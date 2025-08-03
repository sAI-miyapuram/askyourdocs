from fastapi import APIRouter, UploadFile, File, Form
from app.rag_pipeline import ingest_document, process_query_with_rag

router = APIRouter()

@router.post("/ingest-doc")
async def ingest_doc(file: UploadFile = File(...)):
    result = await ingest_document(file)
    return {"status": "success", "details": result}

@router.post("/query")
async def query_docs(query: str = Form(...)):
    answer = await process_query_with_rag(query)
    return {"query": query, "answer": answer}