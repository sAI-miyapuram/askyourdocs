# ui/structured_extractor_app.py
# Streamlit app to upload documents and extract fields -> dynamic JSON via LLM
# Runs locally with Ollama (no OpenAI key). Suppresses noisy warnings.

import warnings
warnings.filterwarnings("ignore")

import io
import json
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st

# Basic loaders (install notes below)
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

try:
    from pdf2image import convert_from_bytes
except Exception:
    convert_from_bytes = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import docx
except Exception:
    docx = None

# LangChain Ollama (no deprecation)
from langchain_ollama import OllamaLLM

# -----------------------
# Config & Helpers
# -----------------------

st.set_page_config(page_title="Structured Field Extractor (RAG+LLM)", layout="wide")
st.title("📄 Any‑Doc → JSON: Field Extractor (Local LLM)")

DEFAULT_MODEL = "llama3"
DEFAULT_BASE_URL = "http://localhost:11434"

SYSTEM_INSTRUCTIONS = """You are a precise information extraction assistant.
Given raw document text from a user-uploaded file, extract key field names and their values.
The output MUST be valid JSON only, no commentary, no markdown.
Adapt the keys to the inferred application/document type (e.g., job application, license renewal, visa form, government application).
- Keep keys human-readable.
- Use ISO date format (YYYY-MM-DD) when possible.
- If a field is present but value is missing, set value to null.
- If you are unsure about the document type, include "Document Type" with your best guess.
- Include a "Confidence" score between 0 and 1 summarizing overall extraction confidence.
Return only JSON.
"""

USER_PROMPT_TEMPLATE = """Document Text (truncated if long):
---
{doc_text}
---

Return only JSON. Example shape (this is only an example; adapt to the doc):
{{
  "Document Type": "License Renewal",
  "Applicant Name": "Jane Smith",
  "License Number": "CA-123456",
  "Expiration Date": "2025-12-31",
  "Address": "123 Main St, San Jose, CA",
  "Confidence": 0.86
}}
"""

MAX_CHARS = 12000  # simple truncation safeguard for very large docs


def clean_json_maybe(s: str) -> Optional[dict]:
    """
    Try to parse JSON from a model response. If it contains extra text,
    attempt to find the largest {...} block.
    """
    s = s.strip()
    # Fast path
    try:
        return json.loads(s)
    except Exception:
        pass

    # Find a JSON object block heuristically
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


def pdf_to_text_with_pdfplumber(file_bytes: bytes) -> str:
    if not pdfplumber:
        return ""
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                text_parts.append(text)
    return "\n".join(text_parts)


def pdf_to_text_with_pypdf2(file_bytes: bytes) -> str:
    if not PdfReader:
        return ""
    text_parts = []
    reader = PdfReader(io.BytesIO(file_bytes))
    for page in reader.pages:
        try:
            text_parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(text_parts)


def pdf_to_text_with_ocr(file_bytes: bytes) -> str:
    """OCR for scanned PDFs: render pages to images, run Tesseract."""
    if not (convert_from_bytes and pytesseract and Image):
        return ""
    text_parts = []
    images = convert_from_bytes(file_bytes)  # list of PIL Images
    for img in images:
        text = pytesseract.image_to_string(img) or ""
        if text.strip():
            text_parts.append(text)
    return "\n".join(text_parts)


def image_to_text(file_bytes: bytes) -> str:
    if not (Image and pytesseract):
        return ""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return pytesseract.image_to_string(img) or ""


def docx_to_text(file_bytes: bytes) -> str:
    if not docx:
        return ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    d = docx.Document(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)
    return "\n".join(p.text for p in d.paragraphs)


def txt_to_text(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def extract_text_any(file_name: str, file_bytes: bytes, ocr_fallback: bool = True) -> Tuple[str, str]:
    """
    Return (ext, extracted_text). ext is normalized like 'pdf','jpg','png','docx','txt'
    """
    suffix = Path(file_name).suffix.lower().strip(".")
    ext = suffix

    if suffix in ("pdf",):
        # Try native text
        text = pdf_to_text_with_pdfplumber(file_bytes) or pdf_to_text_with_pypdf2(file_bytes)
        if not text and ocr_fallback:
            text = pdf_to_text_with_ocr(file_bytes)
        return "pdf", text

    if suffix in ("png", "jpg", "jpeg", "tiff", "bmp", "webp"):
        return "image", image_to_text(file_bytes)

    if suffix in ("docx",):
        return "docx", docx_to_text(file_bytes)

    if suffix in ("txt", "md"):
        return "txt", txt_to_text(file_bytes)

    # Unknown: attempt OCR as image, else treat as text
    maybe_text = image_to_text(file_bytes)
    return (ext or "bin"), maybe_text


def llm_extract_json(doc_text: str, model: str, base_url: str) -> dict:
    # Truncate doc_text to avoid overlong contexts
    doc_text = (doc_text or "").strip()
    if len(doc_text) > MAX_CHARS:
        doc_text = doc_text[:MAX_CHARS] + "\n[TRUNCATED]"

    prompt = USER_PROMPT_TEMPLATE.format(doc_text=doc_text)
    llm = OllamaLLM(model=model, base_url=base_url)

    # A light system/instruction prefix:
    final_input = f"{SYSTEM_INSTRUCTIONS}\n\n{prompt}"

    raw = llm.invoke(final_input)
    parsed = clean_json_maybe(raw) or {"error": "Model did not return valid JSON", "raw": raw}
    return parsed


# -----------------------
# UI
# -----------------------

with st.sidebar:
    st.header("⚙️ Settings")
    base_url = st.text_input("Ollama Endpoint", value=DEFAULT_BASE_URL)
    model = st.text_input("Model", value=DEFAULT_MODEL, help="Any local Ollama model, e.g., llama3, mistral, phi4, qwen2.5")
    ocr_fallback = st.checkbox("Use OCR fallback for PDFs", value=True)
    show_raw = st.checkbox("Show raw LLM text (debug)", value=False)

st.write("Upload **any** document: PDF (native or scanned), images, DOCX, or TXT. The app will extract fields and values to JSON based on inferred document type.")

uploaded_files = st.file_uploader(
    "Upload one or more files",
    type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    all_results = []
    for f in uploaded_files:
        st.divider()
        st.subheader(f"📄 {f.name}")

        file_bytes = f.read()
        ext, text = extract_text_any(f.name, file_bytes, ocr_fallback=ocr_fallback)

        col1, col2 = st.columns([3, 2])
        with col1:
            if text and text.strip():
                st.caption(f"Extracted text preview ({ext}):")
                st.text_area(" ", text[:4000], height=200, key=f"ta_{f.name}")
            else:
                st.warning("No text extracted. Try enabling OCR fallback (sidebar) or check Tesseract installation.")

        if text and text.strip():
            if st.button(f"Extract Fields → JSON ({f.name})"):
                with st.spinner("Analyzing with local LLM…"):
                    result = llm_extract_json(text, model=model, base_url=base_url)

                if show_raw and isinstance(result, dict) and "raw" in result:
                    st.code(result["raw"][:4000], language="json")

                if isinstance(result, dict) and "raw" in result and "error" in result:
                    st.error("Model returned non‑JSON; showing best-effort parse.")
                    st.code(result.get("raw", "")[:4000])

                # Display parsed JSON
                if isinstance(result, dict):
                    st.success("✅ Extracted JSON")
                    st.json(result)

                    # Download
                    dl = json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")
                    st.download_button(
                        "⬇️ Download JSON",
                        data=dl,
                        file_name=f"{Path(f.name).stem}_fields.json",
                        mime="application/json",
                        use_container_width=True
                    )

                all_results.append({"file": f.name, "json": result})

    if all_results:
        st.divider()
        st.caption("Batch complete.")
