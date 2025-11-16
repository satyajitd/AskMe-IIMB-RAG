import os
from utils import env, constants

from langchain_ollama.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

class LLM:
    """
    LLM wrapper that supports multiple providers (Ollama, Gemini).
    Provider is selected via LLM_PROVIDER environment variable.
    """
    def __new__(cls, max_tokens: int = None):
        provider = os.getenv(env.LLM_PROVIDER, "ollama").lower()
        
        # Get max tokens from env or use provided value
        if max_tokens is None:
            max_tokens = int(os.getenv(env.MAX_OUTPUT_TOKENS))
        
        if provider == constants.GEMINI:
            return cls._create_gemini(max_tokens)
        elif provider == constants.OLLAMA_CLOUD:
            return cls._create_ollama_cloud(max_tokens)
        else:  # default to local ollama
            return cls._create_ollama(max_tokens)
    
    @staticmethod
    def _create_ollama(max_tokens: int):
        """Create Ollama LLM instance"""
        model = os.getenv(env.OLLAMA_MODEL)
        base_url = os.getenv(env.OLLAMA_BASE_URL)

        return ChatOllama(
            model=model,
            base_url=base_url,
            num_predict=max_tokens
        )
    
    @staticmethod
    def _create_ollama_cloud(max_tokens: int):
        """Create Ollama Cloud LLM instance"""
        model = os.getenv(env.OLLAMA_MODEL)
        base_url = os.getenv(env.OLLAMA_BASE_URL)
        api_key = os.getenv(env.OLLAMA_CLOUD_API_KEY)

        headers = {
            "Authorization": f"Bearer {api_key}",
        }

        return ChatOllama(
            model=model,
            base_url=base_url,
            num_predict=max_tokens,
            client_kwargs={"headers": headers}
        )
    
    @staticmethod
    def _create_gemini(max_tokens: int):
        """Create Google Gemini LLM instance"""
        model = os.getenv(env.GEMINI_MODEL)
        api_key = os.getenv(env.GEMINI_API_KEY)
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required when using Gemini provider")
        
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            max_output_tokens=max_tokens,
            temperature=0  # Keep consistent with Ollama behavior
        )