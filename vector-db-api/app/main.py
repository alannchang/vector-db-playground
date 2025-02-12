from fastapi import FastAPI
from .routes import documents

app = FastAPI(
    title="Vector Database API",
    description="API for managing and searching documents using vector embeddings",
    version="0.1.0",
)

app.include_router(documents.router, prefix="/api/v1", tags=["documents"])


@app.get("/")
async def root():
    return {"status": "ok", "message": "Vector Database API is running"}
