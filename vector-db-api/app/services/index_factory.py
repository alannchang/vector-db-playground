from typing import Dict, Any, Optional
import logging
import faiss
import numpy as np
from app.models.search import IndexType
from app.config.settings import settings


class IndexFactory:
    """Factory for creating and managing different FAISS indices"""

    def __init__(self):
        self.indices: Dict[str, Any] = {}
        self.dimension: Optional[int] = None
        self.default_index_type = IndexType.HNSW  # Set a sensible default

    def initialize(self, dimension: int):
        """Initialize the index factory with the vector dimension"""
        self.dimension = dimension
        logging.info("Initializing index factory with dimension %d", dimension)

        # Create indices based on configuration
        enabled_indices = settings.ENABLED_INDICES

        if IndexType.FLAT in enabled_indices:
            self.indices[IndexType.FLAT] = faiss.IndexFlatL2(dimension)
            logging.info("Initialized FLAT index")

        if IndexType.HNSW in enabled_indices:
            try:
                hnsw = faiss.IndexHNSWFlat(dimension, 32)
                hnsw.hnsw.efConstruction = 40
                hnsw.hnsw.efSearch = 50
                self.indices[IndexType.HNSW] = hnsw
                logging.info("Initialized HNSW index")
            except Exception as e:
                logging.error("Failed to initialize HNSW index: %s", str(e))

        if IndexType.IVF in enabled_indices:
            try:
                self.indices[IndexType.IVF] = faiss.IndexIVFFlat(
                    faiss.IndexFlatL2(dimension), dimension, 100
                )
                logging.info("Initialized IVF index")
            except Exception as e:
                logging.error("Failed to initialize IVF index: %s", str(e))

        if IndexType.PQ in enabled_indices:
            try:
                # Find a suitable M value that divides the dimension
                for m in [96, 64, 48, 32, 24, 16, 12, 8, 4]:
                    if dimension % m == 0:
                        self.indices[IndexType.PQ] = faiss.IndexPQ(
                            dimension, m, 8  # Use calculated M value
                        )
                        logging.info("Initialized PQ index with M=%d", m)
                        break
                else:
                    logging.warning(
                        "Could not find suitable M for PQ index with dimension %d",
                        dimension,
                    )
            except Exception as e:
                logging.error("Failed to initialize PQ index: %s", str(e))

        if IndexType.IVF_PQ in enabled_indices:
            try:
                # Find a suitable M value that divides the dimension
                for m in [96, 64, 48, 32, 24, 16, 12, 8, 4]:
                    if dimension % m == 0:
                        self.indices[IndexType.IVF_PQ] = faiss.IndexIVFPQ(
                            faiss.IndexFlatL2(dimension), dimension, 100, m, 8
                        )
                        logging.info("Initialized IVF_PQ index with M=%d", m)
                        break
                else:
                    logging.warning(
                        "Could not find suitable M for IVF_PQ index with dimension %d",
                        dimension,
                    )
            except Exception as e:
                logging.error("Failed to initialize IVF_PQ index: %s", str(e))

        if IndexType.SQ in enabled_indices:
            try:
                self.indices[IndexType.SQ] = faiss.IndexScalarQuantizer(
                    dimension, faiss.ScalarQuantizer.QT_8bit
                )
                logging.info("Initialized SQ index")
            except Exception as e:
                logging.error("Failed to initialize SQ index: %s", str(e))

        # Set the default index type
        self.default_index_type = settings.DEFAULT_INDEX_TYPE

        logging.info("Available indices: %s", list(self.indices.keys()))

    def get_index(self, index_type: Optional[IndexType] = None) -> Any:
        """Get an index by type, falling back to default if not specified"""
        if not self.indices:
            raise ValueError("Index factory not initialized")

        # Use specified index or default
        index_type = index_type or self.default_index_type

        # Fall back to FLAT if requested index not available
        if index_type not in self.indices:
            logging.warning(
                "Index type %s not available, using %s", index_type, IndexType.FLAT
            )
            index_type = IndexType.FLAT

        return self.indices[index_type], index_type

    def add_vectors(self, vectors: np.ndarray, index_type: Optional[IndexType] = None):
        """Add vectors to a specific index or all indices"""
        if not self.indices:
            raise ValueError("Index factory not initialized")

        if index_type:
            # Add to specific index
            if index_type not in self.indices:
                raise ValueError(f"Index type {index_type} not available")

            index = self.indices[index_type]
            self._add_to_index(index, vectors)
        else:
            # Add to all indices
            for _idx_type, index in self.indices.items():
                self._add_to_index(index, vectors)

    def _add_to_index(self, index: Any, vectors: np.ndarray):
        """Add vectors to an index, training if necessary"""
        if hasattr(index, "is_trained") and not index.is_trained:
            if hasattr(index, "train"):
                # Check if this is an IVF-based index
                is_ivf = isinstance(index, faiss.IndexIVF)

                # For IVF indices, we need at least as many vectors as clusters
                if is_ivf:
                    n_vectors = vectors.shape[0]
                    n_clusters = index.nlist if hasattr(index, "nlist") else 100

                    if n_vectors < n_clusters:
                        logging.warning(
                            "Not enough vectors (%d) to train IVF index with %d clusters. "
                            "Skipping training for now.",
                            n_vectors,
                            n_clusters,
                        )
                        return  # Skip adding to this index for now

                # Train the index
                try:
                    index.train(vectors)
                except RuntimeError as e:
                    logging.error("Failed to train index: %s", str(e))
                    return  # Skip adding to this index

        # Add vectors to the index
        try:
            index.add(vectors)
        except Exception as e:
            logging.error("Failed to add vectors to index: %s", str(e))

    def search(
        self,
        query_vector: np.ndarray,
        k: int,  # Number of results to return
        index_type: Optional[IndexType] = None,  # Type of index to use
        search_params: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Search for similar vectors using the specified index"""
        if not self.indices:
            raise ValueError("Index factory not initialized")

        index, actual_index_type = self.get_index(index_type)

        # Apply search parameters if provided
        if search_params:
            self._apply_search_params(index, actual_index_type, search_params)

        # Perform search
        distances, indices = index.search(query_vector, k)

        return distances, indices, actual_index_type

    def _apply_search_params(
        self, index: Any, index_type: IndexType, params: Dict[str, Any]
    ):
        """Apply algorithm-specific search parameters"""
        if (
            index_type == IndexType.HNSW
            and hasattr(index, "hnsw")
            and "ef_search" in params
        ):
            index.hnsw.efSearch = params["ef_search"]

        if (
            index_type in [IndexType.IVF, IndexType.IVF_PQ]
            and hasattr(index, "nprobe")
            and "nprobe" in params
        ):
            index.nprobe = params["nprobe"]
