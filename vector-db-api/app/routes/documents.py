from fastapi import APIRouter, HTTPException, Query
from typing import List
from ..models.document import Document, DocumentCreate, DocumentUpdate, SearchQuery
import uuid
from ..services.vector_store import VectorStore
import numpy as np

router = APIRouter()

# Initialize the service
vector_store = VectorStore()


@router.post("/documents/", response_model=Document)
async def create_document(document: DocumentCreate):
    """Create a new document and store its vector embedding."""
    # Create a new document with a random ID
    doc_id = str(uuid.uuid4())

    # Mock embedding for now (replace with actual embedding generation later)
    mock_embedding = np.random.rand(384).astype(np.float32)

    # Add vector to store with document ID
    vector_store.add_vector(doc_id, mock_embedding)

    new_document = Document(
        id=doc_id,
        content=document.content,
        metadata=document.metadata,
        embedding=mock_embedding.tolist(),  # Convert numpy array to list for JSON
        embedding_model="mock-model",
    )
    return new_document


@router.get("/documents/{doc_id}", response_model=Document)
async def get_document(doc_id: str):
    """Retrieve a document by its ID."""
    raise HTTPException(status_code=501, detail="Not implemented")


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
