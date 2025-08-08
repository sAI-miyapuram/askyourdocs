# ui/chat_app.py
# Streamlit chat UI for AskYourDocs (upload PDFs, build FAISS, chat with local LLM via Ollama)
# - Uses langchain_ollama.OllamaLLM (no deprecations)
# - Uses .invoke() API
# - Shows sources + keeps chat history
# - Can persist/load FAISS index

# --- make project root importable (so `from app...` works) ---
from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parents[1]  # .../askyourdocs
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
# ------------------------------------------------------------

import io
import os
import json
import tempfile
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Reuse your project components
from app.extractor import load_and_split
from app.embedder import embeddings
from app.config import get_config


# =========================
# UI config / sidebar
# =========================
st.set_page_config(page_title="AskYourDocs — Chat with your PDFs", layout="wide")
st.title("💬 AskYourDocs — Chat with your documents")

with st.sidebar:
    st.header("⚙️ Settings")
    cfg = get_config()
    default_base = cfg.get("OLLAMA_ENDPOINT", "http://localhost:11434")
    base_url = st.text_input("Ollama Endpoint", value=default_base)
    model = st.text_input("Model", value="llama3", help="Any local Ollama model (e.g., llama3, mistral)")
    top_k = st.slider("Retriever top_k", min_value=2, max_value=10, value=4)
    persist_index = st.checkbox("Persist index to ./faiss_index", value=False)
    show_sources = st.checkbox("Show sources", value=True)
    clear_chat = st.button("🧹 Clear chat history")

# =========================
# Session state
# =========================
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "qa" not in st.session_state:
    st.session_state.qa = None
if "messages" not in st.session_state:
    st.session_state.messages = []

if clear_chat:
    st.session_state.messages = []
    st.experimental_rerun()


# =========================
# Helpers
# =========================
def build_or_extend_index(pdf_files: list[tuple[str, bytes]]):
    """Create/extend FAISS from (filename, bytes) pairs and wire retriever + QA chain."""
    docs_all = []
    for fname, b in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(b)
            tmp_path = tmp.name
        try:
            chunks = load_and_split(tmp_path)
            # keep filename in metadata for citations
            for c in chunks:
                c.metadata = {**(c.metadata or {}), "source": fname}
            docs_all.extend(chunks)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    if not docs_all:
        return

    if st.session_state.vector_store is None:
        st.session_state.vector_store = FAISS.from_documents(docs_all, embedding=embeddings)
    else:
        st.session_state.vector_store.add_documents(docs_all)

    if persist_index:
        st.session_state.vector_store.save_local("faiss_index")

    st.session_state.retriever = st.session_state.vector_store.as_retriever(
        search_kwargs={"k": top_k}
    )
    llm = OllamaLLM(base_url=base_url, model=model)
    st.session_state.qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=st.session_state.retriever,
        return_source_documents=True
    )


def try_load_persisted():
    try:
        vs = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        st.session_state.vector_store = vs
        st.session_state.retriever = vs.as_retriever(search_kwargs={"k": top_k})
        llm = OllamaLLM(base_url=base_url, model=model)
        st.session_state.qa = RetrievalQA.from_chain_type(
            llm=llm, retriever=st.session_state.retriever, return_source_documents=True
        )
        st.success("Loaded persisted index from ./faiss_index")
    except Exception as e:
        st.info(f"No persisted index found (or failed to load): {e}")


def extract_answer_text(output) -> str:
    """RetrievalQA may return str or dict; normalize to text."""
    if isinstance(output, dict):
        return (
            output.get("result")
            or output.get("output_text")
            or output.get("answer")
            or json.dumps(output, ensure_ascii=False, indent=2)
        )
    return str(output)


def format_sources(src_docs) -> list[str]:
    out = []
    for d in src_docs or []:
        name = d.metadata.get("source") or "unknown.pdf"
        snippet = (d.page_content or "").replace("\n", " ")[:280]
        out.append(f"*{name}*: {snippet}…")
    return out


def bulletize(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # leave numbered lists, prefix other lines with dashes
    bullets = []
    for ln in lines:
        if ln[:2].isdigit() or ln.startswith(tuple(f"{i}." for i in range(1, 10))):
            bullets.append(ln)
        else:
            bullets.append(f"- {ln}")
    return "\n".join(bullets) if bullets else text


# =========================
# Upload area
# =========================
st.subheader("📤 Upload PDFs")
uploaded_files = st.file_uploader(
    "Upload one or more PDFs (you can add more later).",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    pairs = [(f.name, f.read()) for f in uploaded_files]
    with st.spinner("Indexing documents…"):
        build_or_extend_index(pairs)
    st.success(f"Indexed {len(uploaded_files)} file(s). You can start chatting below.")

# Button to load persisted index
if st.session_state.vector_store is None and st.button("Load persisted index (./faiss_index)"):
    try_load_persisted()

# =========================
# Chat UI
# =========================
st.subheader("💬 Chat")

# history
for m in st.session_state.messages:
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.markdown(m["content"])
        if m.get("sources") and show_sources and m["role"] == "assistant":
            with st.expander("Sources"):
                for i, s in enumerate(m["sources"], 1):
                    st.markdown(f"**{i}.** {s}")

# input
prompt = st.chat_input("Ask a question about your uploaded documents…")

if prompt:
    if st.session_state.qa is None:
        st.warning("Please upload PDFs or load a persisted index first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                raw = st.session_state.qa.invoke(prompt)  # returns dict with sources
                answer_text = extract_answer_text(raw)
                st.markdown("#### 🧠 Answer")
                st.markdown(bulletize(answer_text))

                src_docs = raw.get("source_documents") if isinstance(raw, dict) else []
                sources_fmt = format_sources(src_docs) if show_sources else []
                if show_sources and sources_fmt:
                    with st.expander("Sources"):
                        for i, s in enumerate(sources_fmt, 1):
                            st.markdown(f"**{i}.** {s}")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer_text, "sources": sources_fmt if show_sources else []}
        )
