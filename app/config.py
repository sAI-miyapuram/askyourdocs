# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()

'''def get_config():
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
        "PINECONE_ENVIRONMENT": os.getenv("PINECONE_ENVIRONMENT")
    }'''

def get_config():
    return {
        "OLLAMA_ENDPOINT": os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
    }

