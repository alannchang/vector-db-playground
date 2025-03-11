from typing import List, Dict, Any, Optional
import uuid
import logging
import numpy as np
from app.services.index_factory import IndexFactory
from app.services.embedding import EmbeddingService
from app.models.search import IndexType, SearchParams

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, embedding_service: EmbeddingService):
        # Initialize as a dictionary to store both IDs and vectors
        self.index_factory = IndexFactory()
        self.embedding_service = embedding_service
        self.document_store: Dict[str, Dict[str, Any]] = {}  # id -> document data
        self.vector_data: List[np.ndarray] = []  # All vectors
        self.id_to_index: Dict[str, int] = {}  # Map document ID to vector index

        logger.info("Initialized VectorStore with SentenceTransformer model")

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        index_type: Optional[IndexType] = None,
    ) -> str:
        """Add a document to the vector store"""
        vector = self.embedding_service.encode([content]).astype("float32")

        # Initialize index factory if this is the first document
        if not self.vector_data:
            self.index_factory.initialize(vector.shape[1])

        self.document_store[doc_id] = {
            "content": content,
            "metadata": metadata or {},
        }

        vector_idx = len(self.vector_data)
        self.vector_data.append(vector[0])
        self.id_to_index[doc_id] = vector_idx

        # Add to index
        self.index_factory.add_vectors(vector, index_type)

        return doc_id

    def search(
        self,
        query: str,
        k: int = 5,
        index_type: Optional[IndexType] = None,
        search_params: Optional[SearchParams] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if not self.vector_data:
            return []

        # Convert search params to dict if provided
        params_dict = search_params.dict() if search_params else None

        # Generate query embedding
        query_vector = self.embedding_service.encode([query]).astype("float32")

        # Search
        distances, indices, used_index_type = self.index_factory.search(
            query_vector, k, index_type, params_dict
        )

        # Format results
        results = []
        for i in range(min(k, len(indices[0]))):
            idx = indices[0][i]
            if idx < 0:  # FAISS returns -1 for not enough results
                continue

            # Find document ID from index
            doc_id = None
            for id, index in self.id_to_index.items():
                if index == idx:
                    doc_id = id
                    break

            if doc_id and doc_id in self.document_store:
                doc = self.document_store[doc_id]
                results.append(
                    {
                        "document_id": doc_id,
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "distance": float(distances[0][i]),
                    }
                )

        return results, used_index_type

    def add_documents_batch(
        self, documents: List[Dict[str, Any]], index_type: Optional[IndexType] = None
    ) -> List[str]:
        """Add multiple documents in a batch"""
        if not documents:
            return []

        contents = [doc["content"] for doc in documents]
        vectors = self.embedding_service.encode(contents).astype("float32")

        # Initialize index factory if this is the first batch
        if not self.vector_data:
            self.index_factory.initialize(vectors.shape[1])

        # Store documents and vectors
        doc_ids = []
        start_idx = len(self.vector_data)

        for i, doc in enumerate(documents):
            doc_id = doc.get("id", str(uuid.uuid4()))
            self.document_store[doc_id] = {
                "content": doc["content"],
                "metadata": doc.get("metadata", {}),
            }
            self.id_to_index[doc_id] = start_idx + i
            doc_ids.append(doc_id)

        # Add vectors to index
        self.vector_data.extend([v for v in vectors])

        # Add to index - batch addition is better for training
        try:
            self.index_factory.add_vectors(vectors, index_type)
        except Exception as e:
            logging.error("Error adding vectors to index: %s", str(e))
            # Continue anyway - we've stored the vectors in vector_data

        return doc_ids
