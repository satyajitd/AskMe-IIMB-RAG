import os
import logging
from logging.handlers import RotatingFileHandler

from utils import env

import chromadb
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

class VectorStore(Chroma):
    def __init__(self):
        # Configure ChromaDB
        self.collection_name = os.getenv(env.CHROMA_COLLECTION_NAME)
        self.client = chromadb.HttpClient(host=os.getenv(env.CHROMA_SERVER_HOST), port=os.getenv(env.CHROMA_SERVER_PORT)) # Point to your Docker-hosted server

        super().__init__(
            client=self.client,
            collection_name=self.collection_name,
        )
        # Configure logging
        self.configure_logging()
    
    def retrieve(self, query: str, n_results: int = 5) -> list[Document]:
        self.logger.info(f"Retrieving top {n_results} documents for the query: {query}")
        results = super().similarity_search(query, k=n_results)
        self.logger.info(f"Retrieved {len(results)} documents.")
        return results

    def configure_logging(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.log_file = os.getenv(env.APP_LOG)
        
        # Configure handlers only if not already present to avoid duplicate logs
        if not self.logger.handlers:
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            # Stream handler (console)
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)
            # Optional file handler
            try:
                file_handler = RotatingFileHandler(self.log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except Exception:
                # If file handler can't be created, log a warning to console
                stream_handler.setLevel(logging.WARNING)
                self.logger.warning("Could not create log file handler at %s", self.log_file)
        self.logger.setLevel(level=logging.INFO)

vector_store = VectorStore()
