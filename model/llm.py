import os
from utils import env

from langchain_ollama.chat_models import ChatOllama

class LLM(ChatOllama):
    def __init__(self, max_tokens: int = None):
        model = os.getenv(env.OLLAMA_MODEL)
        base_url = os.getenv(env.OLLAMA_BASE_URL)
        
        # Get max tokens from env or use provided value
        if max_tokens is None:
            max_tokens = int(os.getenv(env.MAX_OUTPUT_TOKENS, "512"))
        
        super().__init__(
            model=model, 
            base_url=base_url,
            num_predict=max_tokens  # Ollama parameter for max output tokens
        )