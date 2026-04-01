import streamlit as st
import numpy as np
import json
import os
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VectorVault",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg: #08090d;
    --surface: #0f111a;
    --surface2: #161926;
    --border: #1e2235;
    --accent: #7c5cfc;
    --accent2: #00e5ff;
    --accent3: #ff4fd8;
    --text: #e8eaf6;
    --muted: #6b7280;
    --success: #00e676;
    --danger: #ff4444;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Main background */
.stApp {
    background: var(--bg) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text) !important;
}

/* Headings */
h1, h2, h3, h4 { color: var(--text) !important; }

/* Custom header */
.vv-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 24px 0 8px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 28px;
}
.vv-logo {
    font-size: 36px;
    line-height: 1;
}
.vv-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 50%, var(--accent3) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0; padding: 0;
    line-height: 1;
}
.vv-subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 4px;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin: 16px 0;
}
.metric-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.8rem;
    color: var(--accent2);
    margin-top: 4px;
}

/* Upload zone */
.upload-zone {
    border: 2px dashed var(--border);
    border-radius: 16px;
    padding: 32px 24px;
    text-align: center;
    background: var(--surface);
    transition: all 0.3s ease;
    margin-bottom: 16px;
}
.upload-zone:hover {
    border-color: var(--accent);
    background: rgba(124, 92, 252, 0.05);
}

/* Buttons */
.stButton > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
    background: var(--surface2) !important;
    color: var(--text) !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.5px !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    background: rgba(124, 92, 252, 0.15) !important;
    color: var(--accent2) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(124, 92, 252, 0.2) !important;
}

/* Primary button */
div[data-testid="column"]:first-child .stButton > button,
.primary-btn > button {
    background: linear-gradient(135deg, var(--accent), #9c27b0) !important;
    border: none !important;
    color: white !important;
}

/* Text inputs */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124, 92, 252, 0.2) !important;
}

/* Slider */
.stSlider > div > div > div > div {
    background: var(--accent) !important;
}

/* Select box */
.stSelectbox > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* Result cards */
.result-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 22px;
    margin: 10px 0;
    position: relative;
    transition: all 0.2s ease;
}
.result-card:hover {
    border-color: var(--accent);
    transform: translateX(4px);
}
.result-rank {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--accent3);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 6px;
}
.result-source {
    font-weight: 700;
    font-size: 0.95rem;
    color: var(--accent2);
    margin-bottom: 8px;
}
.result-text {
    font-size: 0.88rem;
    color: var(--muted);
    line-height: 1.6;
}
.result-distance {
    margin-top: 10px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.dist-bar-bg {
    flex: 1;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
}
.dist-bar-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.dist-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--accent2);
    white-space: nowrap;
}

/* Section titles */
.section-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 3px;
    margin: 24px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* Doc list item */
.doc-item {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 0.85rem;
}
.doc-icon { font-size: 1.1rem; }
.doc-name { color: var(--text); flex: 1; }
.doc-chunks {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--accent);
    background: rgba(124, 92, 252, 0.1);
    padding: 2px 8px;
    border-radius: 20px;
}

/* Alerts */
.alert-success {
    background: rgba(0, 230, 118, 0.1);
    border: 1px solid rgba(0, 230, 118, 0.3);
    border-radius: 10px;
    padding: 12px 16px;
    color: var(--success);
    font-size: 0.9rem;
    margin: 8px 0;
}
.alert-info {
    background: rgba(0, 229, 255, 0.08);
    border: 1px solid rgba(0, 229, 255, 0.2);
    border-radius: 10px;
    padding: 12px 16px;
    color: var(--accent2);
    font-size: 0.9rem;
    margin: 8px 0;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: var(--muted) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border-bottom: none !important;
}

/* Number input */
.stNumberInput > div > div > input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
}

/* File uploader */
.stFileUploader > div {
    background: var(--surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
}
.stFileUploader > div:hover {
    border-color: var(--accent) !important;
}

/* Progress */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
    border-radius: 4px !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* Plotly container */
.js-plotly-plot { border-radius: 16px; overflow: hidden; }

/* Label colors */
.stTextInput label, .stTextArea label, .stSelectbox label,
.stSlider label, .stNumberInput label, .stFileUploader label {
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
}

</style>
""", unsafe_allow_html=True)

# ── Vector Store ──────────────────────────────────────────────────────────────
STORE_PATH = Path("vector_store.pkl")

def load_store():
    if STORE_PATH.exists():
        with open(STORE_PATH, "rb") as f:
            return pickle.load(f)
    return {"documents": [], "embeddings": [], "metadata": []}

def save_store(store):
    with open(STORE_PATH, "wb") as f:
        pickle.dump(store, f)

def simple_embed(text: str, dim: int = 128) -> np.ndarray:
    """Deterministic text embedding using character n-gram frequencies."""
    text = text.lower().strip()
    vec = np.zeros(dim)
    chars = list(text)
    for i, c in enumerate(chars):
        h = int(hashlib.md5((c + str(i % 5)).encode()).hexdigest(), 16)
        vec[h % dim] += 1
    for i in range(len(chars) - 1):
        bigram = chars[i] + chars[i+1]
        h = int(hashlib.md5(bigram.encode()).hexdigest(), 16)
        vec[h % dim] += 1.5
    words = text.split()
    for w in words:
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        vec[h % dim] += 3
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

def cosine_distance(a, b):
    sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    return 1.0 - sim

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def extract_text_from_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".txt") or name.endswith(".md"):
        return uploaded_file.read().decode("utf-8", errors="replace")
    elif name.endswith(".pdf"):
        try:
            import pdfplumber
            import io
            text_parts = []
            with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text_parts.append(t)
            return "\n\n".join(text_parts)
        except ImportError:
            return uploaded_file.read().decode("utf-8", errors="replace")
    else:
        return uploaded_file.read().decode("utf-8", errors="replace")

# ── Session State ─────────────────────────────────────────────────────────────
if "store" not in st.session_state:
    st.session_state.store = load_store()
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

store = st.session_state.store

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="vv-header">
  <div class="vv-logo">🔮</div>
  <div>
    <div class="vv-title">VectorVault</div>
    <div class="vv-subtitle">Semantic Vector Space Explorer</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── METRICS ───────────────────────────────────────────────────────────────────
n_docs = len(set(m["source"] for m in store["metadata"])) if store["metadata"] else 0
n_chunks = len(store["documents"])
n_dims = len(store["embeddings"][0]) if store["embeddings"] else 128

st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card">
    <div class="metric-label">Documents</div>
    <div class="metric-value">{n_docs}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Vector Chunks</div>
    <div class="metric-value">{n_chunks}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Dimensions</div>
    <div class="metric-value">{n_dims}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 20px 0;">
      <div style="font-family:'Syne',sans-serif; font-weight:800; font-size:1.1rem; 
           background:linear-gradient(135deg,#7c5cfc,#00e5ff); -webkit-background-clip:text;
           -webkit-text-fill-color:transparent;">⬡ Upload Documents</div>
      <div style="font-size:0.72rem; color:#6b7280; font-family:'Space Mono',monospace; 
           margin-top:4px; letter-spacing:1px;">PDF · TXT · MD</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    chunk_size = st.slider("Chunk size (words)", 50, 500, 200, 50)
    overlap = st.slider("Overlap (words)", 0, 100, 30, 10)

    if uploaded_files:
        if st.button("⬡ Vectorize & Store", use_container_width=True):
            progress = st.progress(0)
            status = st.empty()
            total = len(uploaded_files)

            for idx, f in enumerate(uploaded_files):
                source = f.name
                existing = [m["source"] for m in store["metadata"]]
                if source in existing:
                    status.markdown(f'<div class="alert-info">↩ Already stored: {source}</div>', unsafe_allow_html=True)
                    progress.progress((idx + 1) / total)
                    continue

                status.markdown(f'<div class="alert-info">⚙ Processing {source}…</div>', unsafe_allow_html=True)
                text = extract_text_from_file(f)
                chunks = chunk_text(text, chunk_size, overlap)

                for chunk in chunks:
                    emb = simple_embed(chunk)
                    store["documents"].append(chunk)
                    store["embeddings"].append(emb)
                    store["metadata"].append({
                        "source": source,
                        "timestamp": datetime.now().isoformat(),
                        "chunk_len": len(chunk.split()),
                    })

                progress.progress((idx + 1) / total)

            save_store(store)
            st.session_state.store = store
            status.markdown(f'<div class="alert-success">✓ Vectorization complete!</div>', unsafe_allow_html=True)
            st.rerun()

    st.markdown("---")

    # Stored documents list
    if store["metadata"]:
        st.markdown("""<div style="font-family:'Space Mono',monospace; font-size:0.68rem; 
                    color:#6b7280; text-transform:uppercase; letter-spacing:2px; 
                    margin-bottom:12px;">Stored Documents</div>""", unsafe_allow_html=True)

        doc_chunks = {}
        for m in store["metadata"]:
            doc_chunks[m["source"]] = doc_chunks.get(m["source"], 0) + 1

        for doc, cnt in doc_chunks.items():
            ext = doc.split(".")[-1].upper()
            icon = "📄" if ext == "PDF" else "📝"
            short = doc[:22] + "…" if len(doc) > 25 else doc
            st.markdown(f"""
            <div class="doc-item">
              <span class="doc-icon">{icon}</span>
              <span class="doc-name">{short}</span>
              <span class="doc-chunks">{cnt} chunks</span>
            </div>
            """, unsafe_allow_html=True)

    if store["documents"]:
        st.markdown("---")
        if st.button("🗑 Clear Vector Store", use_container_width=True):
            store = {"documents": [], "embeddings": [], "metadata": []}
            save_store(store)
            st.session_state.store = store
            st.rerun()

# ── MAIN TABS ─────────────────────────────────────────────────────────────────
tab_search, tab_visualize, tab_data = st.tabs(["🔍  Search", "🌐  3D Visualize", "📋  Browse"])

# ── TAB: SEARCH ───────────────────────────────────────────────────────────────
with tab_search:
    st.markdown('<div class="section-title">Semantic Search</div>', unsafe_allow_html=True)

    col_q, col_n = st.columns([4, 1])
    with col_q:
        query = st.text_input("Search query", placeholder="Enter keywords or a sentence to find semantically similar content…",
                              label_visibility="collapsed")
    with col_n:
        top_n = st.number_input("Top N", min_value=1, max_value=50, value=5)

    filter_doc = st.selectbox(
        "Filter by document (optional)",
        ["All documents"] + list(set(m["source"] for m in store["metadata"])) if store["metadata"] else ["All documents"],
        label_visibility="visible"
    )

    search_clicked = st.button("⬡ Search Vector Space", use_container_width=True)

    if search_clicked and query.strip():
        if not store["embeddings"]:
            st.markdown('<div class="alert-info">⚠ No vectors stored yet. Upload documents first.</div>', unsafe_allow_html=True)
        else:
            q_emb = simple_embed(query)
            results = []
            for i, (doc, emb, meta) in enumerate(zip(store["documents"], store["embeddings"], store["metadata"])):
                if filter_doc != "All documents" and meta["source"] != filter_doc:
                    continue
                dist = cosine_distance(q_emb, emb)
                results.append({"rank": 0, "text": doc, "distance": dist, "meta": meta, "idx": i})

            results.sort(key=lambda x: x["distance"])
            results = results[:top_n]
            for i, r in enumerate(results):
                r["rank"] = i + 1

            st.session_state.search_results = results
            st.session_state.last_query = query

    if st.session_state.search_results:
        st.markdown(f'<div class="section-title">{len(st.session_state.search_results)} Results for &nbsp;<span style="color:#7c5cfc">"{st.session_state.last_query}"</span></div>', unsafe_allow_html=True)

        max_dist = max(r["distance"] for r in st.session_state.search_results) + 0.001
        min_dist = min(r["distance"] for r in st.session_state.search_results)

        for r in st.session_state.search_results:
            pct = (r["distance"] - min_dist) / (max_dist - min_dist + 0.001)
            sim_pct = round((1 - r["distance"]) * 100, 1)
            bar_pct = round((1 - pct) * 100)
            preview = r["text"][:280] + ("…" if len(r["text"]) > 280 else "")
            src = r["meta"]["source"]
            ts = r["meta"].get("timestamp", "")[:10]

            st.markdown(f"""
            <div class="result-card">
              <div class="result-rank">#{r['rank']} · {src} · {ts}</div>
              <div class="result-text">{preview}</div>
              <div class="result-distance">
                <div class="dist-bar-bg">
                  <div class="dist-bar-fill" style="width:{bar_pct}%"></div>
                </div>
                <div class="dist-label">sim {sim_pct}% · dist {r['distance']:.4f}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

# ── TAB: VISUALIZE ────────────────────────────────────────────────────────────
with tab_visualize:
    st.markdown('<div class="section-title">3D Vector Space</div>', unsafe_allow_html=True)

    if len(store["embeddings"]) < 3:
        st.markdown('<div class="alert-info">⬡ Store at least 3 document chunks to visualize the vector space.</div>', unsafe_allow_html=True)
    else:
        col_method, col_highlight, col_btn = st.columns([2, 2, 1])
        with col_method:
            method = st.selectbox("Reduction method", ["PCA", "t-SNE"], key="viz_method")
        with col_highlight:
            highlight_query = st.text_input("Highlight query point", placeholder="Optional: enter a search term to highlight…", key="viz_query")
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            viz_btn = st.button("⬡ Render", use_container_width=True)

        if viz_btn or (len(store["embeddings"]) > 0):
            embeddings = np.array(store["embeddings"])

            # Add query vector if provided
            query_point = None
            if highlight_query.strip():
                query_vec = simple_embed(highlight_query)
                embeddings_with_q = np.vstack([embeddings, query_vec])
                query_point = len(embeddings)
            else:
                embeddings_with_q = embeddings

            n_samples = embeddings_with_q.shape[0]

            with st.spinner("Computing projection…"):
                if method == "PCA":
                    pca = PCA(n_components=3)
                    coords = pca.fit_transform(embeddings_with_q)
                else:
                    perp = min(30, n_samples - 1)
                    tsne = TSNE(n_components=3, perplexity=perp, random_state=42, max_iter=300)
                    coords = tsne.fit_transform(embeddings_with_q)

            # Build color by source
            sources = [m["source"] for m in store["metadata"]]
            unique_sources = list(set(sources))
            palette = [
                "#7c5cfc", "#00e5ff", "#ff4fd8", "#00e676", "#ffd740",
                "#ff6d00", "#40c4ff", "#ea80fc", "#b9f6ca", "#ff9e80"
            ]
            color_map = {s: palette[i % len(palette)] for i, s in enumerate(unique_sources)}
            colors = [color_map[s] for s in sources]

            fig = go.Figure()

            # Plot each source separately for legend
            for src in unique_sources:
                idxs = [i for i, s in enumerate(sources) if s == src]
                short = src[:20] + "…" if len(src) > 20 else src
                fig.add_trace(go.Scatter3d(
                    x=coords[idxs, 0],
                    y=coords[idxs, 1],
                    z=coords[idxs, 2],
                    mode="markers",
                    name=short,
                    marker=dict(
                        size=5,
                        color=color_map[src],
                        opacity=0.8,
                        line=dict(width=0.5, color="#1e2235"),
                    ),
                    text=[store["documents"][i][:100] + "…" for i in idxs],
                    hovertemplate="<b>%{text}</b><extra>" + short + "</extra>",
                ))

            # Query highlight point
            if query_point is not None:
                fig.add_trace(go.Scatter3d(
                    x=[coords[query_point, 0]],
                    y=[coords[query_point, 1]],
                    z=[coords[query_point, 2]],
                    mode="markers+text",
                    name=f"Query: {highlight_query[:20]}",
                    text=["⬡ QUERY"],
                    textposition="top center",
                    marker=dict(
                        size=14,
                        color="#ffffff",
                        symbol="diamond",
                        line=dict(width=2, color="#ff4fd8"),
                    ),
                    hovertemplate=f"<b>Query: {highlight_query}</b><extra></extra>",
                ))

                # Draw lines from query to top-5 nearest
                dists = [cosine_distance(simple_embed(highlight_query), store["embeddings"][i]) for i in range(len(store["embeddings"]))]
                nearest = np.argsort(dists)[:5]
                for ni in nearest:
                    fig.add_trace(go.Scatter3d(
                        x=[coords[query_point, 0], coords[ni, 0]],
                        y=[coords[query_point, 1], coords[ni, 1]],
                        z=[coords[query_point, 2], coords[ni, 2]],
                        mode="lines",
                        showlegend=False,
                        line=dict(color="rgba(255,79,216,0.35)", width=2),
                        hoverinfo="skip",
                    ))

            fig.update_layout(
                paper_bgcolor="rgba(8,9,13,0)",
                plot_bgcolor="rgba(8,9,13,0)",
                scene=dict(
                    bgcolor="rgba(15,17,26,1)",
                    xaxis=dict(
                        backgroundcolor="rgba(15,17,26,0)",
                        gridcolor="#1e2235",
                        showbackground=True,
                        zerolinecolor="#1e2235",
                        tickfont=dict(color="#6b7280", family="Space Mono", size=9),
                        title="",
                    ),
                    yaxis=dict(
                        backgroundcolor="rgba(15,17,26,0)",
                        gridcolor="#1e2235",
                        showbackground=True,
                        zerolinecolor="#1e2235",
                        tickfont=dict(color="#6b7280", family="Space Mono", size=9),
                        title="",
                    ),
                    zaxis=dict(
                        backgroundcolor="rgba(15,17,26,0)",
                        gridcolor="#1e2235",
                        showbackground=True,
                        zerolinecolor="#1e2235",
                        tickfont=dict(color="#6b7280", family="Space Mono", size=9),
                        title="",
                    ),
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                ),
                legend=dict(
                    bgcolor="rgba(15,17,26,0.9)",
                    bordercolor="#1e2235",
                    borderwidth=1,
                    font=dict(color="#e8eaf6", family="Syne", size=12),
                ),
                margin=dict(l=0, r=0, t=20, b=0),
                height=600,
                font=dict(family="Syne", color="#e8eaf6"),
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            <div class="alert-info" style="margin-top:0;">
              ⬡ &nbsp;Showing <b>{len(store['embeddings'])}</b> vectors across <b>{len(unique_sources)}</b> document(s) · 
              Projection: <b>{method}</b> · 
              Original dims: <b>{store['embeddings'][0].shape[0]}</b> → 3D
            </div>
            """, unsafe_allow_html=True)

# ── TAB: BROWSE ───────────────────────────────────────────────────────────────
with tab_data:
    st.markdown('<div class="section-title">Stored Chunks</div>', unsafe_allow_html=True)

    if not store["documents"]:
        st.markdown('<div class="alert-info">No documents stored yet.</div>', unsafe_allow_html=True)
    else:
        filter_src = st.selectbox(
            "Filter by source",
            ["All"] + list(set(m["source"] for m in store["metadata"])),
            key="browse_filter"
        )

        browse_rows = []
        for i, (doc, meta) in enumerate(zip(store["documents"], store["metadata"])):
            if filter_src != "All" and meta["source"] != filter_src:
                continue
            browse_rows.append((i, doc, meta))

        st.markdown(f'<div style="font-family:Space Mono,monospace;font-size:0.72rem;color:#6b7280;margin-bottom:12px;">{len(browse_rows)} chunks</div>', unsafe_allow_html=True)

        for chunk_i, (i, doc, meta) in enumerate(browse_rows[:100]):
            preview = doc[:200] + ("…" if len(doc) > 200 else "")
            with st.expander(f"#{i}  ·  {meta['source']}  ·  {len(doc.split())} words"):
                st.markdown(f"<div style='font-size:0.88rem;color:#9ca3af;line-height:1.7;'>{doc}</div>", unsafe_allow_html=True)
                st.code(f"Embedding dims: {len(store['embeddings'][i])} | Timestamp: {meta.get('timestamp','N/A')[:19]}", language=None)

        if len(browse_rows) > 100:
            st.markdown(f'<div class="alert-info">Showing first 100 of {len(browse_rows)} chunks.</div>', unsafe_allow_html=True)