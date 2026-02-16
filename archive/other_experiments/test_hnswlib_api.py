"""Quick test of hnswlib proper API."""

import hnswlib
import numpy as np

# Create index
index = hnswlib.Index(space='cosine', dim=3)
index.init_index(max_elements=10, ef_construction=200, M=16)

# Test adding items
vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
ids = np.array([0, 1, 2], dtype=np.int32)

print("Adding vectors...")
index.add_items(vectors, ids)

print("Searching...")
query = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
labels, distances = index.knn_query(query, k=3)

print(f"Labels: {labels}")
print(f"Distances: {distances}")
print("✓ API works!")
