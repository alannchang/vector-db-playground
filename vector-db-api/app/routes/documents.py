"""API routes for managing documents and their vector embeddings."""

import logging
import time
import uuid
from typing import List

from fastapi import APIRouter, HTTPException
from sentence_transformers import SentenceTransformer

from app.models.document import Document, DocumentCreate, DocumentUpdate, SearchQuery
from app.services.vector_store import VectorStore

router = APIRouter()

# Initialize the service
vector_store = VectorStore()
start_time = time.time()
model = SentenceTransformer("all-MiniLM-L6-v2")
logging.info("Model loaded in %.2f seconds.", time.time() - start_time)


@router.post("/documents/", response_model=Document)
async def create_document(document: DocumentCreate):
    """Create a new document and store its vector embedding."""
    # Create a new document with a random ID
    doc_id = str(uuid.uuid4())

    new_document = Document(
        id=doc_id,
        content=document.content,
    )

    mock_embedding = model.encode([document.content]).astype("float32")

    # Add vector to store with document ID
    vector_store.add_vector(doc_id, mock_embedding)

    return new_document


@router.get("/documents/{doc_id}", response_model=Document)
async def get_document(doc_id: str):
    """Retrieve a document by its ID."""
    # raise HTTPException(status_code=501, detail="Not implemented")
    print("checking vector store: ", vector_store.vector_data)
    found_document = Document(
        id=doc_id,
        content="document.content",
    )
    return found_document


@router.put("/documents/{doc_id}", response_model=Document)
async def update_document(doc_id: str, document: DocumentUpdate):
    """Update an existing document."""
    # TODO: Implement document update
    raise HTTPException(status_code=501, detail="Not implemented")


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document by its ID."""
    # TODO: Implement document deletion
    raise HTTPException(status_code=501, detail="Not implemented")


@router.post("/documents/search/", response_model=List[Document])
async def search_similar_documents(query: SearchQuery):
    """Search for similar documents using vector similarity."""
    raise HTTPException(status_code=501, detail="Not implemented")


@router.post("/documents/batch/", response_model=List[Document])
async def batch_create_documents(documents: List[DocumentCreate]):
    """Batch create documents with vector embeddings."""
    raise HTTPException(status_code=501, detail="Not implemented")


@router.post("/documents/{doc_id}/reindex")
async def reindex_document(doc_id: str, model: str = None):
    """
    Regenerate the vector embedding for a document.
    Optionally specify a different embedding model.
    """
    raise HTTPException(status_code=501, detail="Not implemented")
