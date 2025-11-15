import os

from utils import env
from utils.logger import configure_logger

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

class VectorStore(Chroma):
    def __init__(self):
        # Configure ChromaDB
        self.n_results = int(os.getenv(env.CHROMA_N_RESULTS))
        self.collection_name = os.getenv(env.CHROMA_COLLECTION_NAME)
        self.client = chromadb.HttpClient(host=os.getenv(env.CHROMA_SERVER_HOST), port=os.getenv(env.CHROMA_SERVER_PORT)) # Point to your Docker-hosted server
        self.embedding_function = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        super().__init__(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function
        )
        # Configure logging
        self.configure_logging()
    
    def retrieve(self, query: str) -> list[Document]:
        self.logger.info(f"Retrieving top {self.n_results} documents for the query: {query}")
        results = super().similarity_search(query, k=self.n_results)
        self.logger.info(f"Retrieved {len(results)} documents.")
        return results

    def configure_logging(self):
        self.logger = configure_logger(self.__class__.__name__)

vector_store = VectorStore()
