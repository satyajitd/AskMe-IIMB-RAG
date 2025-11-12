import logging
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

import chromadb
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

class VectorStore(Chroma):
    def __init__(self, collection_name: str):
        # Configure ChromaDB
        self.collection_name = collection_name
        self.client = chromadb.HttpClient(host='localhost', port=8000) # Point to your Docker-hosted server
        self.embedding_function = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        super().__init__(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
        )
        # Configure logging
        self.configure_logging()

    def test_connection(self):
        """Tests the connection to the ChromaDB server."""
        self.logger.info("Testing connection to ChromaDB...")
        try:
            collections = self.client.list_collections()
            self.logger.info(f"Successfully connected. Found collections: {collections}")
        except Exception as e:
            self.logger.error(f"Error connecting to ChromaDB: {e}")

    def store(self, documents) -> list[str]:
        self.logger.info(f"Creating embeddings for {len(documents)} documents.")
        embeddings = super().add_documents(documents)
        self.logger.info("Embeddings created successfully.")
        return embeddings
    
    def retrieve(self, query: str, n_results: int = 5) -> list[Document]:
        self.logger.info(f"Retrieving top {n_results} documents for the query: {query}")
        results = super().similarity_search(query, k=n_results)
        self.logger.info(f"Retrieved {len(results)} documents.")
        return results

    def configure_logging(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.log_dir = Path.cwd() / "log"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"vectorstore_{datetime.now().timestamp()}.log"
        
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

vector_store = VectorStore(collection_name="iimb-rag-docker")
