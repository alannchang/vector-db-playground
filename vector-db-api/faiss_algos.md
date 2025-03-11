# Vector Search Algorithms in FAISS

## Product Quantization (PQ)

**Product Quantization (PQ)** is a compression technique used to efficiently store and search high-dimensional vectors.

### How PQ Works:

1. **Vector Splitting**: The original d-dimensional vector is split into M sub-vectors of dimension d/M
2. **Subspace Quantization**: Each sub-vector is quantized separately using a smaller codebook
3. **Compression**: Instead of storing the full vector, only the quantization codes are stored
4. **Distance Approximation**: During search, distances are computed using lookup tables

### Key Parameters:

- **M**: Number of subquantizers (sub-vectors)
  - Must be a divisor of the vector dimension
  - Higher M = more accurate but slower and more memory
- **nbits**: Bits per subquantizer (typically 8)
  - Controls the size of each subquantizer's codebook (2^nbits)

### Error Message Explained:

```
The dimension of the vector (d) should be a multiple of the number of subquantizers (M)
```

This means your vector dimension (e.g., 768) must be divisible by M (e.g., 16). If not, the algorithm can't evenly split the vector.

## Comparison of FAISS Algorithms

| Algorithm                     | Type        | Description                              | Pros                                                        | Cons                                       | Best For                        |
| ----------------------------- | ----------- | ---------------------------------------- | ----------------------------------------------------------- | ------------------------------------------ | ------------------------------- |
| **Flat (IndexFlatL2)**        | Exact       | Brute-force L2 distance computation      | 100% accurate, No training required                         | Slow for large datasets, High memory usage | Small datasets, Benchmarking    |
| **HNSW (IndexHNSWFlat)**      | Graph-based | Hierarchical Navigable Small World graph | Very fast search, No training required                      | High memory usage                          | Speed-critical applications     |
| **IVF (IndexIVFFlat)**        | Clustering  | Inverted file with coarse quantization   | Good balance of speed/accuracy, Configurable via nprobe     | Requires training, Less accurate than Flat | Medium-sized datasets           |
| **PQ (IndexPQ)**              | Compression | Product Quantization                     | Very memory efficient                                       | Lower accuracy, Requires training          | Memory-constrained applications |
| **IVF+PQ (IndexIVFPQ)**       | Hybrid      | Combines IVF and PQ                      | Extremely memory efficient, Scalable to billions of vectors | Lower accuracy, Requires training          | Very large datasets             |
| **SQ (IndexScalarQuantizer)** | Compression | Scalar Quantization                      | Better accuracy than PQ, Memory efficient                   | Less compression than PQ                   | Balance of accuracy and memory  |

## Detailed Algorithm Descriptions

### 1. Flat Index (IndexFlatL2)

The simplest index that computes exact Euclidean (L2) distances between the query and all vectors in the database.

```python
index = faiss.IndexFlatL2(dimension)
```

- **Time Complexity**: O(n × d) where n is the number of vectors and d is the dimension
- **Memory Usage**: O(n × d × 4) bytes (for 32-bit floats)
- **Use When**: You need exact results and have a small to medium dataset

### 2. HNSW (Hierarchical Navigable Small World)

A graph-based approach that creates a multi-layer graph structure for efficient navigation.

```python
index = faiss.IndexHNSWFlat(dimension, M=32)
index.hnsw.efConstruction = 40  # Build-time quality parameter
index.hnsw.efSearch = 50        # Query-time quality parameter
```

- **Parameters**:
  - **M**: Maximum number of connections per node (16-64 typical)
  - **efConstruction**: Controls index build quality (higher = better quality but slower construction)
  - **efSearch**: Controls search quality (higher = more accurate but slower search)
- **Time Complexity**: O(log(n) × d) for search
- **Memory Usage**: Higher than Flat (additional graph connections)
- **Use When**: You need fast, high-quality approximate search

### 3. IVF (Inverted File Index)

Partitions the vector space into clusters, then only searches within relevant clusters.

```python
index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dimension), dimension, nlist=100)
index.train(training_vectors)  # Requires training data
index.nprobe = 10  # Number of clusters to search
```

- **Parameters**:
  - **nlist**: Number of clusters (Voronoi cells)
  - **nprobe**: Number of nearest clusters to search at query time
- **Time Complexity**: O((n/nlist) × nprobe × d) for search
- **Memory Usage**: Similar to Flat plus cluster centroids
- **Use When**: You need a good balance of speed and accuracy

### 4. PQ (Product Quantization)

Compresses vectors by splitting them into subvectors and quantizing each separately.

```python
index = faiss.IndexPQ(dimension, M=16, nbits=8)
index.train(training_vectors)  # Requires training data
```

- **Parameters**:
  - **M**: Number of subquantizers (must divide dimension evenly)
  - **nbits**: Bits per subquantizer (typically 8)
- **Time Complexity**: O(n × M) for search
- **Memory Usage**: O(n × M) bytes (dramatic reduction)
- **Use When**: Memory is severely constrained

### 5. IVF+PQ (Combined Approach)

Combines IVF's space partitioning with PQ's compression for highly efficient search.

```python
index = faiss.IndexIVFPQ(faiss.IndexFlatL2(dimension), dimension, nlist=100, M=16, nbits=8)
index.train(training_vectors)  # Requires training data
index.nprobe = 10  # Number of clusters to search
```

- **Parameters**: Combines parameters from both IVF and PQ
- **Time Complexity**: O((n/nlist) × nprobe × M) for search
- **Memory Usage**: Extremely efficient, scales to billions of vectors
- **Use When**: You have very large datasets and need efficient search

### 6. SQ (Scalar Quantization)

Quantizes each dimension independently, offering a simpler compression scheme than PQ.

```python
index = faiss.IndexScalarQuantizer(dimension, faiss.ScalarQuantizer.QT_8bit)
```

- **Parameters**:
  - **qtype**: Quantization type (8-bit, 4-bit, etc.)
- **Time Complexity**: Similar to Flat but with faster distance computations
- **Memory Usage**: O(n × d × bits/8) bytes
- **Use When**: You want a simple compression scheme with good accuracy

## Implementation Recommendations

1. **For PQ and IVF+PQ**:

   - Ensure your vector dimension is divisible by M
   - For a dimension of 768 (common in BERT models), use M values like 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, or 768
   - For a dimension of 384 (common in smaller models), use M values like 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, or 384

2. **Dynamic M Selection**:

   ```python
   # Calculate appropriate M value based on dimension
   def get_suitable_m(dimension):
       # Try common values that might divide the dimension evenly
       for m in [96, 64, 48, 32, 24, 16, 12, 8]:
           if dimension % m == 0:
               return m
       # Fallback to 1 (not ideal but will work)
       return 1

   M = get_suitable_m(dimension)
   index = faiss.IndexPQ(dimension, M, 8)
   ```

3. **Algorithm Selection Guide**:

   - **< 1M vectors**: HNSW or Flat
   - **1M-10M vectors**: IVF or HNSW
   - **10M-100M vectors**: IVF+PQ
   - **> 100M vectors**: IVF+PQ with careful parameter tuning

4. **Memory vs. Accuracy Tradeoff**:
   - More memory → better accuracy: Flat > HNSW > IVF > SQ > PQ > IVF+PQ
   - Less memory → worse accuracy: IVF+PQ < PQ < SQ < IVF < HNSW < Flat
