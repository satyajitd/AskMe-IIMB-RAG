from langchain_ollama.chat_models import ChatOllama

class LLM(ChatOllama):
    def __init__(self, model: str = "llama3:8b-instruct-q4_K_M", base_url: str = "http://localhost:11434"):
        super().__init__(model=model, base_url=base_url)

__all__ = ["LLM"]