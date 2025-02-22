import numpy as np
from typing import List, Tuple, Dict
import logging
import faiss

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self):
        # Initialize as a dictionary to store both IDs and vectors
        self.vector_data: Dict[str, np.ndarray] = {}
        self.index = None
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Initialized VectorStore")

    def add_vector(self, vector_id: str, vector: np.ndarray) -> None:
        """Add a vector to the store."""
        # Store vector with its ID
        self.vector_data[vector_id] = vector[0]  # Remove batch dimension
        logger.debug(f"Added vector {vector_id} to store")
        self._update_index()

    def get_vector(self, vector_id: str) -> np.ndarray:
        """Retrieve a vector from the store."""
        return self.vector_data.get(vector_id)

    def _update_index(self) -> None:
        """Update the index with all vectors."""
        if not self.vector_data:
            return

        vectors = np.stack(list(self.vector_data.values()))
        dimension = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(vectors)

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
