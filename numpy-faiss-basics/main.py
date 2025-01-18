import logging
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Start logging
logging.info("Starting the vector database implementation.")

# Load a pre-trained model
start_time = time.time()
model = SentenceTransformer("all-MiniLM-L6-v2")
logging.info("Model loaded in %.2f seconds.", time.time() - start_time)

# Example list of documents
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# Convert documents to vectors
start_time = time.time()
batch_size = 32
data = []
for i in range(0, len(documents), batch_size):
    batch = documents[i : i + batch_size]
    data.append(model.encode(batch).astype("float32"))
data = np.vstack(data)
logging.info(
    "Documents encoded into vectors in %.2f seconds.", time.time() - start_time
)

# Create a FAISS index
start_time = time.time()
dimension = data.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(data)
logging.info(
    "FAISS index created and vectors added in %.2f seconds.", time.time() - start_time
)

# Query with a new document
query_document = "This is a new document."
query_vector = model.encode([query_document]).astype("float32")

# Search for the 2 nearest neighbors
start_time = time.time()
k = 2
distances, indices = index.search(query_vector, k)
logging.info("Search completed in %.2f seconds.", time.time() - start_time)

# Print results
print("Distances:", distances)
print("Indices:", indices)

# Translate indices to original documents
for i in range(k):
    print(f"Nearest neighbor {i + 1}:")
    print(f"Document: {documents[indices[0][i]]}")
    print(f"Distance: {distances[0][i]}")
