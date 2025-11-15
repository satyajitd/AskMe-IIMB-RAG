import os
from typing import List

from utils import env
from utils.logger import configure_logger

import chromadb
from langchain_core.documents import Document

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.schema import NodeWithScore


class VectorStore:
    def __init__(self):
        # Configure logging
        self.configure_logging()
        
        # Configure ChromaDB client
        self.n_results = int(os.getenv(env.CHROMA_N_RESULTS))
        self.collection_name = os.getenv(env.CHROMA_COLLECTION_NAME)
        
        chroma_client = chromadb.HttpClient(
            host=os.getenv(env.CHROMA_SERVER_HOST),
            port=int(os.getenv(env.CHROMA_SERVER_PORT))
        )
        
        # Get or create collection
        chroma_collection = chroma_client.get_or_create_collection(self.collection_name)
        
        # Configure LlamaIndex embedding model  
        embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")
        
        # Set global embedding model for LlamaIndex
        Settings.embed_model = embed_model
        
        # Initialize vector store and index
        self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=storage_context
        )
        
        self.logger.info(
            f"VectorStore initialized with Nomic embeddings via Ollama, "
            f"collection: {self.collection_name}, top_k: {self.n_results}"
        )
    
    def retrieve(self, query: str) -> List[Document]:
        """Retrieve relevant documents using LlamaIndex retriever."""
        self.logger.info(f"Retrieving top {self.n_results} documents for query: {query}")
        
        # Create retriever from index
        retriever = self.index.as_retriever(similarity_top_k=self.n_results)
        
        # Retrieve nodes
        nodes: List[NodeWithScore] = retriever.retrieve(query)
        
        # Convert LlamaIndex nodes to LangChain Documents for workflow compatibility
        documents = [
            Document(
                page_content=node.node.get_content(),
                metadata=node.node.metadata or {}
            )
            for node in nodes
        ]
        
        self.logger.info(f"Retrieved {len(documents)} documents.")
        return documents

    def configure_logging(self):
        self.logger = configure_logger(self.__class__.__name__)


vector_store = VectorStore()
