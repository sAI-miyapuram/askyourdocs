# 🧠 AskYourDocs

A local-first, open-source document question-answering system built with LangChain, FAISS, HuggingFace Embeddings, and LLaMA 3 via Ollama.

> Upload your PDFs. Ask anything. Get answers with citations — 100% offline and private.

---

## 🚀 Features

- 📄 Upload one or many PDF documents
- 💬 Ask natural language questions
- ⚙️ Fast, local AI answers using LLaMA 3 (Ollama)
- 🧠 Document understanding with vector search (FAISS)
- 🧪 Built-in answer evaluation (TruLens-ready)
- 🖥️ Streamlit web UI for interactive experience
- 🔧 API endpoints (FastAPI backend)
- 🐳 Docker-ready deployment

---

## 🏗️ Tech Stack

- **Frontend:** Streamlit
- **Backend:** FastAPI
- **Vector Store:** FAISS
- **Embeddings:** HuggingFace (MiniLM)
- **LLM:** LLaMA 3 via Ollama
- **PDF Parsing:** LangChain Community Loaders

---

## 🛠️ Getting Started

```bash
# 1. Install Ollama & start LLaMA 3
ollama run llama3

# 2. Install dependencies
pip install -r requirements.txt

# 3. Ingest PDFs
python -m scripts.ingest

# 4. Launch Streamlit UI
streamlit run ui/app.py
