from app.extractor import load_and_split
from app.embedder import embeddings
from app.llm_generator import get_llm
from app.evaluator import evaluate_response
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from fastapi import HTTPException
import tempfile
import os

vector_store = None

async def ingest_document(file):
    global vector_store
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Load and split PDF into text chunks
        chunks = load_and_split(tmp_path)
        os.remove(tmp_path)

        if not chunks:
            raise ValueError("No text chunks could be extracted from the uploaded PDF.")

        # Embed and index the chunks
        vector_store = FAISS.from_documents(chunks, embedding=embeddings)
        return f"Ingested {len(chunks)} chunks."

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

async def process_query_with_rag(query):
    if vector_store is None:
        raise HTTPException(status_code=400, detail="No document ingested yet. Please call /ingest-doc first.")

    try:
        retriever = vector_store.as_retriever()
        llm = get_llm()
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        result = qa.run(query)
        evaluation = evaluate_response(query, result)
        return {"answer": result, "evaluation": evaluation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
