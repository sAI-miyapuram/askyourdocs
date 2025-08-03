from fastapi import FastAPI
from app.endpoints import router

app = FastAPI(title="AskYourDocs RAG API")

app.include_router(router)

@app.get("/")
def read_root():
    return {"message": "Welcome to AskYourDocs API"}
