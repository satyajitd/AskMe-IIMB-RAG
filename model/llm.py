import os
from utils import env

from langchain_ollama.chat_models import ChatOllama

class LLM(ChatOllama):
    def __init__(self):
        self.model = os.getenv(env.OLLAMA_MODEL)
        self.base_url = os.getenv(env.OLLAMA_BASE_URL)
        super().__init__(model=self.model, base_url=self.base_url)