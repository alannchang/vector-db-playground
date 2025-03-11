from typing import List
import logging
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config.settings import settings


class EmbeddingService:
    """Service for generating text embeddings"""

    def __init__(self):
        self.model = None

    def initialize(self):
        """Load the embedding model"""
        logging.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
        start_time = time.time()
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logging.info("Model loaded in %.2f seconds", time.time() - start_time)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if self.model is None:
            self.initialize()

        return self.model.encode(texts)
