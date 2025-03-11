from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStore

# Singleton instances
embedding_service = EmbeddingService()
embedding_service.initialize()  # Initialize at startup
vector_store = VectorStore(embedding_service)


def get_embedding_service():
    return embedding_service


def get_vector_store():
    return vector_store
