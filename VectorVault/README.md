# 🔮 VectorVault — Semantic Vector Space Explorer

A beautiful Streamlit app for uploading documents, vectorizing them, performing semantic search, and exploring the vector space in interactive 3D.

---

## ✨ Features

- **Upload** PDF, TXT, or MD files via the sidebar
- **Vectorize** documents into semantic embedding chunks (configurable chunk size & overlap)
- **Search** using natural language — returns top-N results with cosine similarity scores and visual distance bars
- **Visualize** the full vector space in 3D using PCA or t-SNE projection, with optional query point highlighting and nearest-neighbor lines
- **Browse** all stored chunks with expandable detail view
- **Persistent storage** — your vectors survive between sessions (saved to `vector_store.pkl`)

---

## 🚀 Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 📁 File Structure

```
vector_space_app/
├── app.py               ← Main Streamlit application
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
└── vector_store.pkl     ← Auto-created when you store vectors
```

---

## 🧠 How Embedding Works

The app uses a **custom deterministic character/word n-gram embedding** (no external API needed):
- Character unigrams and bigrams from text
- Word-level hashing into a 128-dimensional space
- L2-normalized vectors
- Cosine distance for similarity search

For production use, you can swap `simple_embed()` in `app.py` with any embedding model (e.g. `sentence-transformers`, OpenAI embeddings, etc.)

---

## 💡 Tips

- Use **chunk size 200–300 words** for best retrieval granularity
- **t-SNE** gives better cluster separation in 3D; **PCA** is faster
- Type a query in the "Highlight query point" field to see it placed in 3D space with lines to its nearest neighbors
- Filter search results by document to compare sources