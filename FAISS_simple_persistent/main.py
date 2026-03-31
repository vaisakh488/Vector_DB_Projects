import faiss
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer

# File paths
INDEX_FILE = "faiss_index.bin"
DOC_FILE = "documents.pkl"

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

#Sample documents (your "knowledge base")
documents = [
    "This is a simple Program to demonstrate FAISS vector storage.",
    "There are 10 apples in the basket",
    "Cricket is the favorite sport in India ",
    "The capital of India is New Delhi",
    "There are 7 continents in the world",
    "The largest country in the world is Russia"
]
# -------------------------------
# STEP 1: Check if index exists
# -------------------------------

if os.path.exists(INDEX_FILE) and os.path.exists(DOC_FILE):
    print("🔄 Loading existing FAISS index...")

    index = faiss.read_index(INDEX_FILE)

    with open(DOC_FILE, "rb") as f:
        documents = pickle.load(f)

else:
    print("⚙️ Creating new FAISS index...")

    # Convert documents to embeddings
    doc_embeddings = model.encode(documents)
    doc_embeddings = np.array(doc_embeddings).astype("float32")

    # Create FAISS index
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings
    index.add(doc_embeddings)

    # Save index
    faiss.write_index(index, INDEX_FILE)

    # Save documents
    with open(DOC_FILE, "wb") as f:
        pickle.dump(documents, f)

    print("✅ Index created and saved!")

# -------------------------------
# STEP 2: Query
# -------------------------------

query = input("\nAsk a question: ")

query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")

k = 2
distances, indices = index.search(query_embedding, k)

print("\nTop matches:\n")

for i, idx in enumerate(indices[0]):
    print(f"{i+1}. {documents[idx]} (Distance: {distances[0][i]:.4f})")