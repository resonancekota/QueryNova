from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

import base64
import io
import uuid
import fitz  # PyMuPDF
from PIL import Image
import numpy as np

# =========================================================
# OCR ENGINE
# =========================================================

OCR_ENGINE = None
OCR_SUPPORT = False

try:
    from rapidocr_onnxruntime import RapidOCR

    try:
        OCR_ENGINE = RapidOCR(lang="en")
        print("RapidOCR loaded (lang=en).")
    except TypeError:
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
client = OpenAI()

conversations = {}

MAX_CHARS = 120000
MAX_HISTORY_MESSAGES = 40

OCR_DPI = 300
VISION_DPI = 200
VISION_FALLBACK = True


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# CONVERSATION MANAGEMENT
# =========================================================

@app.route("/new-chat", methods=["POST"])
def new_chat():
    cid = str(uuid.uuid4())

    conversations[cid] = {
        "messages": [],
        "document": None
    }

    return jsonify({
        "conversation_id": cid
    })


def get_conversation(cid=None):
    if cid and cid in conversations:
        return cid

    new_id = str(uuid.uuid4())

    conversations[new_id] = {
        "messages": [],
        "document": None
    }

    return new_id


@app.route("/clear-document", methods=["POST"])
def clear_document():
    """
    Remove the currently uploaded document but keep chat history.
    """
    try:
        data = request.get_json(silent=True) or {}
        cid = data.get("conversation_id")

        cid = get_conversation(cid)

        conversations[cid]["document"] = None

        return jsonify({
            "conversation_id": cid,
            "message": "Document removed."
        })

    except Exception as e:
        print("CLEAR DOCUMENT ERROR:", repr(e))
        return jsonify({"error": str(e)}), 500


# =========================================================
# OCR HELPERS
# =========================================================

def ocr_image_bytes(image_bytes):
    """
    Run RapidOCR on raw image bytes.
    """
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
            line[1]
            for line in result
            if line and len(line) > 1
        )

        print(f"OCR extracted {len(text)} characters")

        return text

    except Exception as e:
        print("OCR ERROR:", repr(e))
        return ""


def ocr_pdf_page(page, dpi=OCR_DPI):
    """
    Render PDF page and run OCR.
    """
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    pix = page.get_pixmap(
        matrix=mat,
        alpha=False
    )

    img_bytes = pix.tobytes("png")

    return ocr_image_bytes(img_bytes)


def vision_extract_page(page, dpi=VISION_DPI):
    """
    Last-resort vision transcription.
    """
    try:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        pix = page.get_pixmap(
            matrix=mat,
            alpha=False
        )

        b64 = base64.b64encode(
            pix.tobytes("png")
        ).decode("utf-8")

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
                            "superscripts, states of matter, "
                            "coefficients and equation arrows. "
                            "Output plain text only."
                        )
                    },
                    {
                        "type": "input_image",
                        "image_url": (
                            f"data:image/png;base64,{b64}"
                        )
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
    Extract text from PDF while attempting to preserve
    subscripts and superscripts.
    """

    raw = page.get_text(
        "dict",
        sort=True
    )

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

                    gap = (
                        span["bbox"][0]
                        - prev["bbox"][2]
                    )

                    dy = abs(
                        span["bbox"][1]
                        - prev["bbox"][1]
                    )

                    is_script = dy > 2

                    if gap > 1.5 and not is_script:
                        parts.append(" ")

                parts.append(txt)

                prev = span

            joined = "".join(parts)

            if joined.strip():
                lines_out.append(joined)

    return "\n".join(lines_out)


def extract_pdf_text(file_bytes):

    if not file_bytes:
        raise RuntimeError(
            "The PDF file is empty."
        )

    try:
        pdf = fitz.open(
            stream=file_bytes,
            filetype="pdf"
        )

    except Exception as e:

        print(
            "PyMuPDF open failed:",
            repr(e)
        )

        pdf = None

    pages_out = []

    if pdf is not None:

        print(
            f"\nPDF pages: {pdf.page_count}"
        )

        for i in range(pdf.page_count):

            page = pdf.load_page(i)

            print(
                f"--- Page {i + 1} ---"
            )

            # -----------------------------------------
            # 1. PDF TEXT LAYER
            # -----------------------------------------

            text = extract_page_text_layer(page)

            print(
                f"  text layer: "
                f"{len(text.strip())} chars"
            )

            # -----------------------------------------
            # 2. OCR ONLY IF NECESSARY
            # -----------------------------------------

            if (
                len(text.strip()) < 40
                and OCR_SUPPORT
            ):

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

                        text = (
                            "[OCR of page]\n"
                            + ocr_text.strip()
                        )

            # -----------------------------------------
            # 3. VISION FALLBACK
            # -----------------------------------------

            if (
                len(text.strip()) < 20
                and VISION_FALLBACK
            ):

                print(
                    "  running vision fallback ..."
                )

                vis_text = vision_extract_page(page)

                if vis_text.strip():

                    if text.strip():

                        text = (
                            text.strip()
                            + "\n\n"
                            "[Vision transcription]\n"
                            + vis_text.strip()
                        )

                    else:

                        text = (
                            "[Vision transcription]\n"
                            + vis_text.strip()
                        )

            if text.strip():

                pages_out.append(
                    f"--- Page {i + 1} ---\n"
                    f"{text.strip()}"
                )

            else:

                print(
                    f"  page {i + 1}: "
                    "no text recovered"
                )

        pdf.close()

    # =====================================================
    # PyPDF2 FALLBACK
    # =====================================================

    if not pages_out and PDF2_SUPPORT:

        try:

            reader = PdfReader(
                io.BytesIO(file_bytes)
            )

            pages = []

            for i, page in enumerate(
                reader.pages,
                start=1
            ):

                t = page.extract_text() or ""

                if t.strip():

                    pages.append(
                        f"--- Page {i} ---\n"
                        f"{t.strip()}"
                    )

            if pages:
                pages_out = pages

        except Exception as e:

            print(
                "PyPDF2 ERROR:",
                repr(e)
            )

    if pages_out:

        return "\n\n".join(
            pages_out
        )

    return None


# =========================================================
# UPLOAD
# =========================================================

@app.route("/upload", methods=["POST"])
def upload():

    try:

        file = request.files.get("file")

        cid = get_conversation(
            request.form.get(
                "conversation_id"
            )
        )

        if not file:
            return jsonify({
                "error": "No file was uploaded."
            }), 400

        filename = file.filename or ""

        if not filename:

            return jsonify({
                "error": "Invalid file name."
            }), 400

        mime_type = (
            file.mimetype
            or "application/octet-stream"
        )

        # =================================================
        # IMAGE
        # =================================================

        if mime_type.startswith("image/"):

            image_bytes = file.read()

            if not image_bytes:

                return jsonify({
                    "error":
                        "The image file is empty."
                }), 400

            image_b64 = base64.b64encode(
                image_bytes
            ).decode("utf-8")

            ocr_text = ocr_image_bytes(
                image_bytes
            )

            conversations[cid]["document"] = {

                "type": "image",

                "filename": filename,

                "base64": image_b64,

                "mime_type": mime_type,

                "ocr_text": ocr_text
            }

            preview = (
                ocr_text
                or "Image uploaded successfully."
            )[:1000]

            return jsonify({

                "conversation_id": cid,

                "filename": filename,

                "text": preview
            })

        # =================================================
        # PDF
        # =================================================

        if (
            mime_type == "application/pdf"
            or filename.lower().endswith(".pdf")
        ):

            pdf_bytes = file.read()

            print(
                f"\nRECEIVED PDF: "
                f"{filename} "
                f"({len(pdf_bytes)} bytes)"
            )

            extracted = extract_pdf_text(
                pdf_bytes
            )

            if not extracted:

                return jsonify({
                    "error": (
                        "The PDF was uploaded, "
                        "but no text could be "
                        "extracted."
                    )
                }), 400

            if len(extracted) > MAX_CHARS:

                extracted = (
                    extracted[:MAX_CHARS]
                    + "\n\n[PDF truncated]"
                )

            conversations[cid]["document"] = {

                "type": "pdf",

                "filename": filename,

                "text": extracted
            }

            preview = extracted[:1000]

            if len(extracted) > 1000:
                preview += "..."

            print(
                f"Total extracted: "
                f"{len(extracted)} chars\n"
            )

            return jsonify({

                "conversation_id": cid,

                "filename": filename,

                "text": preview
            })

        return jsonify({
            "error":
                f"Unsupported file type: "
                f"{mime_type}."
        }), 400

    except Exception as e:

        print(
            "UPLOAD ERROR:",
            repr(e)
        )

        return jsonify({
            "error": str(e)
        }), 500


# =========================================================
# SYSTEM PROMPTS
# =========================================================

GENERAL_INSTRUCTIONS = r"""
You are a helpful scientific assistant.

You can answer questions about:
- Chemistry
- Physics
- Biology
- Mathematics
- Programming
- General knowledge
- Other normal user questions

Answer the user's CURRENT question directly.

Do NOT assume the user is asking about an uploaded
document unless the question clearly refers to it.

If there is no document-related request, answer the
question normally.

For chemistry:

Inline:
\( \ce{H2SO4} \)

Display:
\[ \ce{2H2 + O2 -> 2H2O} \]

Equilibrium:
\[ \ce{N2 + 3H2 <=> 2NH3} \]

Ions:
\( \ce{Fe^{3+}} \)
\( \ce{SO4^{2-}} \)

States:
\[ \ce{CaCO3(s) + 2HCl(aq) -> CaCl2(aq) + H2O(l) + CO2(g)} \]

Never put chemistry inside triple-backtick code blocks.

Use mhchem formatting for chemical formulas and equations.

Be concise and educational.
"""


DOCUMENT_INSTRUCTIONS = r"""
You are a scientific-document assistant.

When the user asks about the uploaded document:

1. Use the document as the primary source.
2. Faithfully transcribe chemical formulas and equations.
3. Repair obvious OCR corruption only when supported
   by the document.
4. Never invent missing formulas.
5. If something is unreadable, explicitly say so.

Chemistry formatting:

Inline:
\( \ce{H2SO4} \)

Display:
\[ \ce{2H2 + O2 -> 2H2O} \]

Equilibrium:
\[ \ce{N2 + 3H2 <=> 2NH3} \]

Ions:
\( \ce{Fe^{3+}} \)

States:
\[ \ce{CaCO3(s) + 2HCl(aq) -> CaCl2(aq) + H2O(l) + CO2(g)} \]

Never put chemistry inside triple-backtick code blocks.

Use \( ... \) for inline chemistry.

Use \[ ... \] for standalone equations.

If the user asks a normal question unrelated to the
document, answer it normally and do not force the
document into the answer.
"""


# =========================================================
# DOCUMENT DETECTION
# =========================================================

def question_is_about_document(query):
    """
    Decide whether the current question is probably
    referring to the uploaded document.

    This intentionally uses simple rules so we do NOT
    make an OpenAI request just to classify every message.
    """

    q = query.lower().strip()

    document_words = [
        "this pdf",
        "this document",
        "the pdf",
        "the document",
        "uploaded file",
        "uploaded pdf",
        "uploaded image",
        "this page",
        "this image",
        "in the pdf",
        "in the document",
        "from the pdf",
        "from the document",
        "according to the pdf",
        "according to the document",
        "page ",
        "above",
        "shown above",
        "extract",
        "transcribe",
        "equations in it",
        "formula in it"
    ]

    for word in document_words:

        if word in q:
            return True

    return False


# =========================================================
# CHAT HISTORY
# =========================================================

def build_history(convo):
    """
    Return only recent conversation messages.

    This prevents the prompt from growing forever.
    """

    messages = convo.get(
        "messages",
        []
    )

    return messages[
        -MAX_HISTORY_MESSAGES:
    ]


# =========================================================
# CHAT
# =========================================================

@app.route("/chat", methods=["POST"])
def chat():

    try:

        data = request.get_json(
            silent=True
        ) or {}

        query = (
            data.get("message", "")
            or ""
        ).strip()

        cid = get_conversation(
            data.get(
                "conversation_id"
            )
        )

        web_search = bool(
            data.get(
                "web_search",
                False
            )
        )

        convo = conversations[cid]

        # -----------------------------------------------
        # Empty message
        # -----------------------------------------------

        if not query:

            query = (
                "Extract and format every "
                "chemical formula and equation "
                "in the document."
            )

        doc = convo.get(
            "document"
        )

        # -----------------------------------------------
        # Decide if document should be used
        # -----------------------------------------------

        document_mode = (
            doc is not None
            and question_is_about_document(
                query
            )
        )

        # -----------------------------------------------
        # Tools
        # -----------------------------------------------

        tools = []

        if web_search:
            tools.append({
                "type": "web_search"
            })

        # =================================================
        # IMAGE + DOCUMENT QUESTION
        # =================================================

        if (
            document_mode
            and doc
            and doc["type"] == "image"
        ):

            mime = doc["mime_type"]

            b64 = doc["base64"]

            ocr_text = (
                doc.get(
                    "ocr_text",
                    ""
                )
                or "(no OCR text available)"
            )

            history = build_history(
                convo
            )

            content = [

                {
                    "type": "input_text",
                    "text": (
                        DOCUMENT_INSTRUCTIONS
                        + "\n\n"
                        "CURRENT USER QUESTION:\n"
                        + query
                        + "\n\n"
                        "OCR TEXT:\n"
                        + ocr_text[:8000]
                    )
                },

                {
                    "type": "input_image",
                    "image_url":
                        f"data:{mime};base64,{b64}"
                }
            ]

            input_messages = []

            for msg in history:

                input_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

            input_messages.append({
                "role": "user",
                "content": content
            })

            response = client.responses.create(

                model="gpt-5.6-luna",

                tools=tools,

                input=input_messages
            )

            answer = response.output_text

        # =================================================
        # PDF + DOCUMENT QUESTION
        # =================================================

        elif (
            document_mode
            and doc
            and doc["type"] == "pdf"
        ):

            pdf_text = doc["text"]

            history = build_history(
                convo
            )

            document_prompt = (
                DOCUMENT_INSTRUCTIONS
                + "\n\n"
                "CURRENT USER QUESTION:\n"
                + query
                + "\n\n"
                "============================\n"
                "DOCUMENT\n"
                "============================\n"
                + pdf_text
                + "\n\n"
                "============================\n"
                "END DOCUMENT\n"
                "============================"
            )

            input_messages = []

            for msg in history:

                input_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

            input_messages.append({
                "role": "user",
                "content": document_prompt
            })

            response = client.responses.create(

                model="gpt-5.6-luna",

                tools=tools,

                input=input_messages
            )

            answer = response.output_text

        # =================================================
        # NORMAL CHAT
        # =================================================

        else:

            history = build_history(
                convo
            )

            input_messages = [

                {
                    "role": "system",
                    "content": GENERAL_INSTRUCTIONS
                }
            ]

            for msg in history:

                input_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

            input_messages.append({
                "role": "user",
                "content": query
            })

            response = client.responses.create(

                model="gpt-5.6-luna",

                tools=tools,

                input=input_messages
            )

            answer = response.output_text

        # =================================================
        # SAVE HISTORY
        # =================================================

        convo["messages"].append({

            "role": "user",

            "content": query
        })

        convo["messages"].append({

            "role": "assistant",

            "content": answer
        })

        # Keep memory bounded
        if len(convo["messages"]) > (
            MAX_HISTORY_MESSAGES * 2
        ):

            convo["messages"] = (
                convo["messages"]
                [-MAX_HISTORY_MESSAGES * 2:]
            )

        return jsonify({

            "conversation_id": cid,

            "answer": answer
        })

    except Exception as e:

        print(
            "CHAT ERROR:",
            repr(e)
        )

        return jsonify({
            "error": str(e)
        }), 500


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )