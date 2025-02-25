import logging
import time
from typing import List, Dict

import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize FastAPI app
app = FastAPI()

# Load the SentenceTransformer model
logging.info("Loading model...")
start_time = time.time()
model = SentenceTransformer("all-MiniLM-L6-v2")
logging.info("Model loaded in %.2f seconds.", time.time() - start_time)

# Data store for tracking added documents
datastore = {}  # Maps index position to text
vector_data = []  # Stores vectors
index = None  # FAISS index, initialized later


# Initialize FAISS index
def initialize_index(dimension: int):
    global index
    index = faiss.IndexFlatL2(dimension)
    logging.info("FAISS index initialized with dimension %d.", dimension)


# Request models
class AddDocumentRequest(BaseModel):
    documents: List[str]


class SearchQueryRequest(BaseModel):
    query: str
    k: int = 2  # Default: find top 2 similar vectors


# Endpoint to add documents
@app.post("/add/")
def add_documents(request: AddDocumentRequest):
    global index, vector_data

    start_time = time.time()
    batch = request.documents
    vectors = model.encode(batch).astype("float32")

    # Initialize index if it's the first time
    if index is None:
        initialize_index(vectors.shape[1])

    # Store text and map to index
    start_idx = len(datastore)
    for i, doc in enumerate(batch):
        datastore[start_idx + i] = doc

    # Add vectors to the FAISS index
    index.add(vectors)
    vector_data.extend(vectors)  # Keep a copy of vectors in memory

    logging.info("Added %d documents in %.2f seconds.", len(batch), time.time() - start_time)
    return {"message": "Documents added successfully.", "count": len(batch)}


# Endpoint to search for similar documents
@app.post("/search/")
def search_similar(request: SearchQueryRequest):
    if index is None or len(datastore) == 0:
        return {"error": "No vectors in the database. Please add documents first."}

    start_time = time.time()
    query_vector = model.encode([request.query]).astype("float32")

    distances, indices = index.search(query_vector, request.k)

    results = []
    for i in range(request.k):
        idx = indices[0][i]
        if idx in datastore:
            results.append({"document": datastore[idx], "distance": float(distances[0][i])})

    logging.info("Search completed in %.2f seconds.", time.time() - start_time)
    return {"query": request.query, "results": results}


# Endpoint to list all stored vectors
@app.get("/documents/")
def get_documents():
    if not datastore:
        return {"message": "No documents stored."}

    return {"documents": list(datastore.values())}
