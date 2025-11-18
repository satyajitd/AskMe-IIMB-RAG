import os
from typing import List

from utils import env, constants
from utils.logger import configure_logger

import weaviate
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.classes.query import MetadataQuery
from langchain_core.documents import Document

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.schema import NodeWithScore
from datetime import datetime, timezone


class VectorStore:
    def __init__(self):
        # Configure logging
        self.configure_logging()
        
        # Configure Weaviate client
        self.n_results = int(os.getenv(env.WEAVIATE_N_RESULTS))
        self.collection_name = os.getenv(env.WEAVIATE_COLLECTION_NAME)
        
        # Connect to Weaviate
        self.client = weaviate.connect_to_local(
            host=os.getenv(env.WEAVIATE_SERVER_HOST),
            port=int(os.getenv(env.WEAVIATE_SERVER_PORT)),
            additional_config=AdditionalConfig(
                timeout=Timeout(init=30, query=60, insert=120)
            )
        )
        
        # Configure LlamaIndex embedding model  
        embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")
        self.embed_model = embed_model
        
        # Set global embedding model for LlamaIndex
        Settings.embed_model = embed_model
        
        # Search configuration
        self.search_mode = os.getenv(env.WEAVIATE_SEARCH_MODE).lower()
        try:
            self.hybrid_alpha = float(os.getenv(env.WEAVIATE_HYBRID_ALPHA))
        except ValueError:
            self.hybrid_alpha = 0.5
        
        # Initialize vector store and index
        self.vector_store = WeaviateVectorStore(
            weaviate_client=self.client,
            index_name=self.collection_name,
            text_key="text"
        )
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=storage_context
        )
        
        self.logger.info(
            f"VectorStore initialized with FastEmbed embeddings, "
            f"collection: {self.collection_name}, top_k: {self.n_results}"
        )
    
    def retrieve(self, query: str) -> List[Document]:
        """Retrieve relevant documents.
        Uses hybrid search when WEAVIATE_SEARCH_MODE=hybrid, else semantic.
        """
        mode = self.search_mode
        if mode == constants.WEAVIATE_SEARCH_MODE_HYBRID:
            self.logger.info(
                f"[Hybrid] Retrieving top {self.n_results} documents (alpha={self.hybrid_alpha}) for query: {query}"
            )
            return self._retrieve_hybrid(query, top_k=self.n_results, alpha=self.hybrid_alpha)
        else:
            self.logger.info(f"[Semantic] Retrieving top {self.n_results} documents for query: {query}")
            # Create retriever from index (pure vector search)
            retriever = self.index.as_retriever(similarity_top_k=self.n_results)
            nodes: List[NodeWithScore] = retriever.retrieve(query)
            documents = [
                Document(
                    page_content=node.node.get_content(),
                    metadata=node.node.metadata or {}
                )
                for node in nodes
            ]
            self.logger.info(f"Retrieved {len(documents)} documents.")
            return documents

    def _retrieve_hybrid(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Document]:
        """Retrieve documents using Weaviate hybrid search (BM25 + vector)."""
        collection = self.client.collections.get(self.collection_name)
        query_embedding = self.embed_model.get_query_embedding(query)
        response = collection.query.hybrid(
            query=query,
            vector=query_embedding,
            alpha=alpha,
            limit=top_k,
            return_metadata=MetadataQuery(score=True, distance=True)
        )

        documents: List[Document] = []
        for obj in response.objects:
            props = obj.properties or {}
            text = props.get("text", "")
            meta = {k: v for k, v in props.items() if k != "text"}
            # include scoring info for downstream consumers
            if obj.metadata and obj.metadata.score is not None:
                meta["_score"] = obj.metadata.score
            if obj.metadata and obj.metadata.distance is not None:
                meta["_distance"] = obj.metadata.distance
            documents.append(Document(page_content=text, metadata=meta))
        self.logger.info(f"Hybrid retrieved {len(documents)} documents.")
        return documents

    def upsert_documents(self, documents: List[Document]) -> int:
        """Insert or update documents into Weaviate with embeddings and metadata.

        Each document's `page_content` is stored under the `text` property. All
        `Document.metadata` fields are persisted as Weaviate properties. A few
        system fields are added when missing: `source`, `ingested_at`, `ingest_via`.

        Returns the number of documents successfully upserted.
        """
        if not documents:
            return 0

        collection = self.client.collections.get(self.collection_name)
        ingested = 0
        now_iso = datetime.now(timezone.utc).isoformat()

        for doc in documents:
            try:
                text = (doc.page_content or "").strip()
                if not text:
                    continue

                meta = dict(doc.metadata or {})
                meta.setdefault("source", meta.get("source", "web"))
                meta.setdefault("ingest_via", "auto-web-search")
                meta["ingested_at"] = now_iso

                vec = self.embed_model.get_text_embedding(text)
                properties = {**meta, "text": text}

                collection.data.insert(properties=properties, vector=vec)
                ingested += 1
            except Exception as exc:
                self.logger.warning("Failed to upsert document: %s", exc)

        self.logger.info("Upserted %d documents into Weaviate", ingested)
        return ingested
    
    def configure_logging(self):
        self.logger = configure_logger(self.__class__.__name__)
    
    def close(self):
        """Close the Weaviate client connection."""
        if self.client:
            self.client.close()
            self.logger.info("Weaviate client connection closed")
    
    def __del__(self):
        """Ensure Weaviate client is closed on object destruction."""
        self.close()

vector_store = VectorStore()
