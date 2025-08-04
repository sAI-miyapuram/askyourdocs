"""
Batch-ingest all PDFs from a folder into a single FAISS index.
Usage:
    python -m scripts.ingest
"""

import os
from pathlib import Path
from app.extractor import load_and_split
from app.embedder import embeddings
from langchain_community.vectorstores import FAISS

DOC_FOLDER = "/data/sample_docs"
INDEX_PATH = "/faiss_index"

def run_batch_ingestion():
    docs_path = Path(DOC_FOLDER)
    if not docs_path.exists():
        raise FileNotFoundError(f" Folder not found: {DOC_FOLDER}")

    pdf_files = list(docs_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f" No PDF files found in {DOC_FOLDER}")

    all_chunks = []
    for pdf_file in pdf_files:
        print(f" Processing: {pdf_file.name}")
        try:
            chunks = load_and_split(str(pdf_file))
            print(f" {len(chunks)} chunks loaded from {pdf_file.name}")
            all_chunks.extend(chunks)
        except Exception as e:
            print(f" Failed to process {pdf_file.name}: {e}")

    if not all_chunks:
        raise ValueError(" No chunks extracted from any PDF.")

    print(f"\n Total chunks: {len(all_chunks)}")
    vector_store = FAISS.from_documents(all_chunks, embedding=embeddings)
    vector_store.save_local(INDEX_PATH)
    print(f" Vector index saved to: {INDEX_PATH}/")

if __name__ == "__main__":
    try:
        run_batch_ingestion()
    except Exception as e:
        print(f"Ingestion failed: {e}")
