from sentence_transformers import SentenceTransformer
import numpy as np

# Sample documents
documents = [
    "Artificial Intelligence is transforming industries.",
    "Machine learning enables computers to learn from data.",
    "Vector databases store embeddings for similarity search.",
    "Endee is a high performance vector database."
]

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
vectors = model.encode(documents)

print("Embeddings created successfully!")
print(vectors)
