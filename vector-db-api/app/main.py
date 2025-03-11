from fastapi import FastAPI
from .routes import documents, search
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Import dependencies to ensure they're initialized
from .dependencies import embedding_service, vector_store  # noqa: F401

app = FastAPI(
    title="Vector Database API",
    description="API for managing and searching documents using vector embeddings",
    version="0.1.0",
)

app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(search.router, prefix="/search", tags=["search"])


@app.get("/")
async def root():
    return {"status": "ok", "message": "Vector Database API is running"}
