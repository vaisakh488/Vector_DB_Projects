import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Sample documents (your "knowledge base")
documents = [
    "This is a simple Program to demonstrate FAISS vector storage.",
    "There are 10 apples in the basket",
    "Cricket is the favorite sport in India ",
    "The capital of India is New Delhi",
    "There are 7 continents in the world",
    "The largest country in the world is Russia"
]

# 3. Convert documents to embeddings
doc_embeddings = model.encode(documents)

# Convert to numpy array (FAISS needs float32)
doc_embeddings = np.array(doc_embeddings).astype("float32")

# 4. Create FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# 5. Add embeddings to index
index.add(doc_embeddings)

print(f"Stored {index.ntotal} documents in FAISS")

# 6. Query
query = input("\nAsk a question: ")

# Convert query to embedding
query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")

# 7. Search
k = 2  # top results
distances, indices = index.search(query_embedding, k)

# 8. Show results
print("\nTop matches:\n")
for i, idx in enumerate(indices[0]):
    print(f"{i+1}. {documents[idx]} (Distance: {distances[0][i]:.4f})")