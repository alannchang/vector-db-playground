import numpy as np
from typing import List, Tuple, Dict, Any
import logging
import faiss
from sentence_transformers import SentenceTransformer
from app.models.document import Document

logger = logging.getLogger(__name__)



class VectorStore:
    def __init__(self):
        # Initialize as a dictionary to store both IDs and vectors
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        # Store both vectors and document content
        self.vector_data: Dict[str, Dict[str, Any]] = {}
        
        # FAISS index
        self.index = None
        
        # Map to track the position of each document in the index
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        
        logger.info("Initialized VectorStore with SentenceTransformer model")

    def add_document(self, doc_id: str, content: str) -> None:
        """Add a document to the vector store with automatic vectorization."""
        vector = self.model.encode([content]).astype("float32")
        
        self.vector_data[doc_id] = {
            "vector": vector[0],
            "content": content
        }
        
        self._update_index()
    
        logger.debug(f"Added document {doc_id} to store")


    def get_document(self, doc_id: str) -> Document:
        """Retrieve a document from the store."""
        if doc_id not in self.vector_data:
            return None

        doc_data = self.vector_data[doc_id]
        return Document(
            id=doc_id,
            content=doc_data["content"]
        )


    def _update_index(self) -> None:
        """Update the index with all vectors."""
        if not self.vector_data:
            return

        # Extract vectors from the stored data
        vectors = np.array([data["vector"] for data in self.vector_data.values()]).astype("float32")

        if self.index is None:
            self.initialize_index(vectors.shape[1])
        
        # Get dimension from the first vector
        dimension = vectors.shape[1]
        
        # Create a new index
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(vectors)
        
        logger.info(f"Updated FAISS index with {len(vectors)} vectors")

    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        # return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    # def find_similar_vectors(
    #     self, query_vector: np.ndarray, num_results: int = 5
    # ) -> List[Tuple[str, float]]:
    #     """Find similar vectors to the query vector."""
    #     results = []
    #     for vector_id, vector in self.vector_data.items():
    #         similarity = self._calculate_similarity(query_vector, vector)
    #         results.append((vector_id, similarity))

    #     results.sort(key=lambda x: x[1], reverse=True)
    #     return results[:num_results]

    def initialize_index(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)
        logging.info("FAISS index initialized with dimension %d.", dimension)