from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

import base64
import io
import uuid
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import os
# =========================================================
# OCR ENGINE
# =========================================================

OCR_ENGINE = None
OCR_SUPPORT = False

try:
    from rapidocr_onnxruntime import RapidOCR
    try:
        # Newer API (v2 / v3)
        OCR_ENGINE = RapidOCR(lang="en")
        print("RapidOCR loaded (lang=en).")
    except TypeError:
        # Older v1.x API - no lang parameter
        OCR_ENGINE = RapidOCR()
        print("RapidOCR loaded (default lang).")
    OCR_SUPPORT = True
except Exception as e:
    print("RapidOCR not available:", repr(e))
    OCR_ENGINE = None
    OCR_SUPPORT = False

try:
    from PyPDF2 import PdfReader
    PDF2_SUPPORT = True
except ImportError:
    PDF2_SUPPORT = False


# =========================================================
# CONFIG
# =========================================================

load_dotenv()

app = Flask(__name__)
# client = OpenAI()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
conversations = {}

MAX_CHARS = 100000
OCR_DPI = 300          # raised from 200 for small fonts / subscripts
VISION_DPI = 200       # dpi for vision-model fallback
VISION_FALLBACK = True # use vision model if OCR finds nothing


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/new-chat", methods=["POST"])
def new_chat():
    cid = str(uuid.uuid4())
    conversations[cid] = {"messages": [], "document": None}
    return jsonify({"conversation_id": cid})


def get_conversation(cid=None):
    if cid and cid in conversations:
        return cid
    new_id = str(uuid.uuid4())
    conversations[new_id] = {"messages": [], "document": None}
    return new_id


# =========================================================
# OCR HELPERS
# =========================================================

def ocr_image_bytes(image_bytes):
    """Run RapidOCR on raw image bytes. Returns text or ''."""
    if not OCR_SUPPORT or OCR_ENGINE is None:
        print("OCR skipped: engine not available")
        return ""

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)
        print(f"OCR running on image {arr.shape} ...")

        result, _ = OCR_ENGINE(arr)

        if not result:
            print("OCR returned empty result")
            return ""

        text = "\n".join(
            line[1] for line in result
            if line and len(line) > 1
        )
        print(f"OCR extracted {len(text)} characters")
        return text

    except Exception as e:
        print("OCR ERROR:", repr(e))
        return ""


def ocr_pdf_page(page, dpi=OCR_DPI):
    """Render a PDF page at high DPI and OCR it."""
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("png")
    return ocr_image_bytes(img_bytes)


def vision_extract_page(page, dpi=VISION_DPI):
    """
    Last-resort fallback: send a rendered page image to the vision model
    and ask it to transcribe everything, preserving chemistry notation.
    Only used when both text extraction AND OCR return nothing.
    """
    try:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")

        response = client.responses.create(
            model="gpt-5.6-luna",
            input=[{
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Transcribe ALL text from this page. "
                            "Preserve chemical formulas, subscripts, "
                            "superscripts, states of matter (s/l/g/aq), "
                            "coefficients, and equation arrows exactly "
                            "as they appear. Output plain text only."
                        )
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{b64}"
                    }
                ]
            }]
        )
        return response.output_text

    except Exception as e:
        print("VISION FALLBACK ERROR:", repr(e))
        return ""


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_page_text_layer(page):
    """
    Extract text using 'dict' mode. Keep subscripts/superscripts glued
    to the previous span by checking vertical position, not just gap.
    """
    raw = page.get_text("dict", sort=True)
    lines_out = []

    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            parts = []
            prev = None

            for span in line.get("spans", []):
                txt = span.get("text", "")
                if not txt:
                    continue

                if prev is not None:
                    gap = span["bbox"][0] - prev["bbox"][2]
                    dy = abs(span["bbox"][1] - prev["bbox"][1])
                    is_script = dy > 2   # sub / superscript
                    if gap > 1.5 and not is_script:
                        parts.append(" ")

                parts.append(txt)
                prev = span

            joined = "".join(parts)
            if joined.strip():
                lines_out.append(joined)

    return "\n".join(lines_out)


def extract_pdf_text(file_bytes):
    """
    For each page:
      1. Try the PDF text layer.
      2. If the text layer is thin (<40 chars), run RapidOCR.
      3. If OCR still returns nothing, call the vision model as a
         last-resort per page.
    """
    if not file_bytes:
        raise RuntimeError("The PDF file is empty.")

    try:
        pdf = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        print("PyMuPDF open failed:", repr(e))
        pdf = None

    pages_out = []

    if pdf is not None:
        print(f"\nPDF pages: {pdf.page_count}")

        for i in range(pdf.page_count):
            page = pdf.load_page(i)
            print(f"--- Page {i+1} ---")

            # 1) text layer
            text = extract_page_text_layer(page)
            print(f"  text layer: {len(text.strip())} chars")

            # 2) OCR if text too short
            if len(text.strip()) < 40 and OCR_SUPPORT:
                print("  running OCR ...")
                ocr_text = ocr_pdf_page(page)
                if ocr_text.strip():
                    if text.strip():
                        text = (
                            text.strip()
                            + "\n\n[OCR of page]\n"
                            + ocr_text.strip()
                        )
                    else:
                        text = "[OCR of page]\n" + ocr_text.strip()

            # 3) Vision fallback if still nothing
            if len(text.strip()) < 20 and VISION_FALLBACK:
                print("  running vision fallback ...")
                vis_text = vision_extract_page(page)
                if vis_text.strip():
                    if text.strip():
                        text = (
                            text.strip()
                            + "\n\n[Vision transcription]\n"
                            + vis_text.strip()
                        )
                    else:
                        text = "[Vision transcription]\n" + vis_text.strip()

            if text.strip():
                pages_out.append(f"--- Page {i+1} ---\n{text.strip()}")
            else:
                print(f"  page {i+1}: no text recovered")

        pdf.close()

    # ---- PyPDF2 fallback if PyMuPDF failed completely ----
    if not pages_out and PDF2_SUPPORT:
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            pages = []
            for i, page in enumerate(reader.pages, start=1):
                t = page.extract_text() or ""
                if t.strip():
                    pages.append(f"--- Page {i} ---\n{t.strip()}")
            if pages:
                pages_out = pages
        except Exception as e:
            print("PyPDF2 ERROR:", repr(e))

    if pages_out:
        return "\n\n".join(pages_out)

    return None


# =========================================================
# UPLOAD
# =========================================================

@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files.get("file")
        cid = get_conversation(request.form.get("conversation_id"))

        if not file:
            return jsonify({"error": "No file was uploaded."}), 400

        filename = file.filename or ""
        if not filename:
            return jsonify({"error": "Invalid file name."}), 400

        mime_type = file.mimetype or "application/octet-stream"

        # ---------------- IMAGE ----------------
        if mime_type.startswith("image/"):
            image_bytes = file.read()
            if not image_bytes:
                return jsonify({"error": "The image file is empty."}), 400

            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            ocr_text = ocr_image_bytes(image_bytes)

            conversations[cid]["document"] = {
                "type": "image",
                "filename": filename,
                "base64": image_b64,
                "mime_type": mime_type,
                "ocr_text": ocr_text,
            }

            preview = (ocr_text or "Image uploaded successfully.")[:1000]
            return jsonify({
                "conversation_id": cid,
                "filename": filename,
                "text": preview,
            })

        # ---------------- PDF ----------------
        if mime_type == "application/pdf" or filename.lower().endswith(".pdf"):
            pdf_bytes = file.read()
            print(f"\nRECEIVED PDF: {filename} ({len(pdf_bytes)} bytes)")

            extracted = extract_pdf_text(pdf_bytes)

            if not extracted:
                return jsonify({
                    "error": (
                        "The PDF was uploaded, but no text could be "
                        "extracted and OCR returned nothing. "
                        "This is likely a corrupted or blank PDF."
                    )
                }), 400

            if len(extracted) > MAX_CHARS:
                extracted = extracted[:MAX_CHARS] + "\n\n[PDF truncated]"

            conversations[cid]["document"] = {
                "type": "pdf",
                "filename": filename,
                "text": extracted,
            }

            preview = extracted[:1000]
            if len(extracted) > 1000:
                preview += "..."

            print(f"Total extracted: {len(extracted)} chars\n")

            return jsonify({
                "conversation_id": cid,
                "filename": filename,
                "text": preview,
            })

        return jsonify({
            "error": f"Unsupported file type: {mime_type}."
        }), 400

    except Exception as e:
        print("UPLOAD ERROR:", repr(e))
        return jsonify({"error": str(e)}), 500


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_INSTRUCTIONS = r"""
You are a scientific-document assistant specialised in chemistry,
physics, biology and mathematics.

Your PRIMARY job when a document or image is provided:
  1. Faithfully TRANSCRIBE the chemical formulas and equations
     found in the source.
  2. Format them with mhchem so they render properly.
  3. Only explain or solve if the user asks.

--------------------------------------------------
CHEMISTRY FORMATTING (mhchem)
--------------------------------------------------
Inline:      \( \ce{H2SO4} \)
Display:     \[ \ce{2H2 + O2 -> 2H2O} \]
Equilibrium: \[ \ce{N2 + 3H2 <=> 2NH3} \]
Ions:        \( \ce{Fe^{3+}} \), \( \ce{SO4^{2-}} \)
States:      \[ \ce{CaCO3(s) + 2HCl(aq) -> CaCl2(aq) + H2O(l) + CO2(g)} \]

Rules:
- NEVER put chemistry inside triple-backtick code blocks.
- NEVER escape backslashes. Write \ce{...}, NOT \\ce{...}.
- Use \(...\) for inline, \[...\] for standalone equations.

--------------------------------------------------
REPAIRING PDF / OCR TEXT
--------------------------------------------------
Extracted text is often mangled. Mentally repair it:

  "H 2 SO 4"           -> \ce{H2SO4}
  "H2SO4 (aq)"         -> \ce{H2SO4(aq)}
  "Fe 3+"              -> \ce{Fe^{3+}}
  "C 6 H 12 O 6"       -> \ce{C6H12O6}
  "N2 + 3H2 <-> 2NH3"  -> \ce{N2 + 3H2 <=> 2NH3}
  "2 H2 + O2 = 2 H2O"  -> \ce{2H2 + O2 -> 2H2O}

Repair ONLY using context actually present in the source.
Never invent a formula that isn't supported by the text/image.
If a formula is unreadable, say so explicitly instead of guessing.

--------------------------------------------------
IMAGE INPUT
--------------------------------------------------
When the user provides an image of a page, textbook, or lab sheet:
- Read every chemical formula and equation you can see.
- Transcribe them exactly, in \ce{...} form.
- Preserve coefficients, charges, states (s/l/g/aq), arrows.
- If a symbol is blurry, mark it as (?) rather than guessing.

--------------------------------------------------
ANSWER STYLE
--------------------------------------------------
- Concise, educational, well-formatted Markdown.
- For every equation, give it on its own line with \[ \ce{...} \].
- Do not wrap the whole answer in a code block.
"""


# =========================================================
# CHAT
# =========================================================

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True) or {}
        query = (data.get("message", "") or "").strip()
        cid = get_conversation(data.get("conversation_id"))
        web_search = bool(data.get("web_search", False))

        convo = conversations[cid]

        if not query:
            query = "Extract and format every chemical formula and equation in the document."

        doc = convo.get("document")
        tools = [{"type": "web_search"}] if web_search else []

        # ---------------- IMAGE ----------------
        if doc and doc["type"] == "image":
            mime = doc["mime_type"]
            b64 = doc["base64"]
            ocr_text = doc.get("ocr_text", "") or "(no OCR text available)"

            user_block = (
                SYSTEM_INSTRUCTIONS
                + "\n\nUSER QUESTION:\n" + query
                + "\n\n=== OCR TEXT (may be incomplete / noisy) ===\n"
                + ocr_text[:8000]
            )

            response = client.responses.create(
                model="gpt-5.6-luna",
                tools=tools,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_block},
                        {"type": "input_image",
                         "image_url": f"data:{mime};base64,{b64}"},
                    ],
                }],
            )
            answer = response.output_text

        # ---------------- PDF ----------------
        elif doc and doc["type"] == "pdf":
            pdf_text = doc["text"]

            prompt = f"""
{SYSTEM_INSTRUCTIONS}

==================================================
USER QUESTION
==================================================
{query}

==================================================
DOCUMENT TEXT (PDF extraction + OCR)
==================================================
{pdf_text}

==================================================
END DOCUMENT
==================================================

Task: Answer the user's question using the document.
If the user asked for equations, transcribe and format EVERY
equation you find, repairing corrupted characters using the
rules above. Never invent equations that are not present.
"""

            response = client.responses.create(
                model="gpt-5.6-luna",
                tools=tools,
                input=prompt,
            )
            answer = response.output_text

        # ---------------- TEXT ----------------
        else:
            prompt = f"{SYSTEM_INSTRUCTIONS}\n\nUSER QUESTION:\n{query}"
            response = client.responses.create(
                model="gpt-5.6-luna",
                tools=tools,
                input=prompt,
            )
            answer = response.output_text

        convo["messages"].append({"role": "user", "content": query})
        convo["messages"].append({"role": "assistant", "content": answer})

        return jsonify({"conversation_id": cid, "answer": answer})

    except Exception as e:
        print("CHAT ERROR:", repr(e))
        return jsonify({"error": str(e)}), 500


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)