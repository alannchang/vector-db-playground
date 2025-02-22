import numpy as np
from typing import List, Tuple, Dict
import logging
import faiss

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self):
        self.vector_data: np.ndarray = []
        logger.info("Initialized VectorStore")

    def add_vector(self, vector_id: str, vector: np.ndarray) -> None:
        """Add a vector to the store."""
        logger.debug("adding to vector store")
        self.vector_data.append(vector)
        print("added to store: ", self.vector_data)
        self.vector_data = np.vstack(self.vector_data)
        # self.vector_data[vector_id] = vector
        self._update_index()
        logger.debug(f"Added vector {vector_id} to store")

    # def get_vector(self, vector_id: str) -> np.ndarray:
    #     """Retrieve a vector from the store."""
    #     return self.vector_data.get(vector_id)

    def _update_index(self) -> None:
        """Update the index with the new vector."""
        dimension = self.vector_data.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(self.vector_data)

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
