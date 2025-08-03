from langchain_community.llms import Ollama
from app.config import get_config

'''def get_llm():
    return OpenAI(temperature=0)'''

def get_llm():
    config = get_config()
    return Ollama(base_url=config["OLLAMA_ENDPOINT"], model="llama3")