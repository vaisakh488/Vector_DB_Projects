import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import umap
import plotly.graph_objects as go

# -------------------------------
# CONFIG
# -------------------------------
METHOD = "umap"   # "pca" | "tsne" | "umap"
TOP_K = 3         # nearest neighbors to highlight

# -------------------------------
# LOAD MODEL
# -------------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------------
# SAMPLE DATA
# -------------------------------
documents = [
    "Python is a programming language",
    "Java is also used in backend systems",
    "Docker is used for containerization",
    "Kubernetes manages containers at scale",
    "FAISS is used for similarity search",
    "Vector databases store embeddings",
    "Machine learning models learn from data",
    "Deep learning uses neural networks",
    "GitHub is used for version control",
    "CI/CD automates deployment pipelines"
]

# -------------------------------
# EMBEDDINGS
# -------------------------------
doc_embeddings = model.encode(documents)
doc_embeddings = np.array(doc_embeddings).astype("float32")
doc_embeddings = normalize(doc_embeddings)

# -------------------------------
# QUERY INPUT
# -------------------------------
query = input("\nEnter your query: ")

query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")
query_embedding = normalize(query_embedding)

# -------------------------------
# SIMILARITY (for highlighting)
# -------------------------------
similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
top_k_idx = np.argsort(similarities)[-TOP_K:]

# -------------------------------
# REDUCER
# -------------------------------
if METHOD == "pca":
    reducer = PCA(n_components=3)

elif METHOD == "tsne":
    reducer = TSNE(n_components=3, perplexity=5, random_state=42)

elif METHOD == "umap":
    reducer = umap.UMAP(n_components=3)

else:
    raise ValueError("Invalid METHOD")

# -------------------------------
# DIMENSION REDUCTION
# -------------------------------
if METHOD == "tsne":
    combined = np.vstack([doc_embeddings, query_embedding])
    reduced_all = reducer.fit_transform(combined)

    reduced_vectors = reduced_all[:-1]
    query_reduced = reduced_all[-1:]

else:
    reduced_vectors = reducer.fit_transform(doc_embeddings)
    query_reduced = reducer.transform(query_embedding)

# -------------------------------
# PREP DATA
# -------------------------------
x = reduced_vectors[:, 0]
y = reduced_vectors[:, 1]
z = reduced_vectors[:, 2]

qx, qy, qz = query_reduced[0]

# Highlight logic
colors = []
sizes = []

for i in range(len(documents)):
    if i in top_k_idx:
        colors.append("red")
        sizes.append(8)
    else:
        colors.append("blue")
        sizes.append(5)

# -------------------------------
# PLOTLY FIGURE
# -------------------------------
fig = go.Figure()

# Documents
fig.add_trace(go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers+text',
    text=documents,
    textposition="top center",
    marker=dict(
        size=sizes,
        color=colors,
        opacity=0.8
    ),
    name="Documents"
))

# Query
fig.add_trace(go.Scatter3d(
    x=[qx],
    y=[qy],
    z=[qz],
    mode='markers+text',
    text=["QUERY"],
    textposition="top center",
    marker=dict(
        size=12,
        color="green",
        symbol="diamond"
    ),
    name="Query"
))

# -------------------------------
# LAYOUT
# -------------------------------
fig.update_layout(
    title=f"3D Vector Space Visualization ({METHOD.upper()})",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z"
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

fig.show()


# --------------------------Using Matplotlib -----------------------------------


# import numpy as np
# import matplotlib.pyplot as plt
# from sentence_transformers import SentenceTransformer
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.preprocessing import normalize
# import umap
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
#
# # -------------------------------
# # CONFIG
# # -------------------------------
# METHOD = "umap"   # "pca" | "tsne" | "umap"
#
# # -------------------------------
# # LOAD MODEL
# # -------------------------------
# model = SentenceTransformer('all-MiniLM-L6-v2')
#
# # -------------------------------
# # SAMPLE DATA
# # -------------------------------
# documents = [
#     "Python is a programming language",
#     "Java is also used in backend systems",
#     "Docker is used for containerization",
#     "Kubernetes manages containers at scale",
#     "FAISS is used for similarity search",
#     "Vector databases store embeddings",
#     "Machine learning models learn from data",
#     "Deep learning uses neural networks",
#     "GitHub is used for version control",
#     "CI/CD automates deployment pipelines"
# ]
#
# # -------------------------------
# # EMBEDDINGS
# # -------------------------------
# doc_embeddings = model.encode(documents)
# doc_embeddings = np.array(doc_embeddings).astype("float32")
#
# # Normalize (important for better similarity representation)
# doc_embeddings = normalize(doc_embeddings)
#
# # -------------------------------
# # QUERY INPUT
# # -------------------------------
# query = input("\nEnter your query: ")
#
# query_embedding = model.encode([query])
# query_embedding = np.array(query_embedding).astype("float32")
# query_embedding = normalize(query_embedding)
#
# # -------------------------------
# # REDUCER SELECTION
# # -------------------------------
# if METHOD == "pca":
#     reducer = PCA(n_components=3)
#
# elif METHOD == "tsne":
#     reducer = TSNE(n_components=3, perplexity=5, random_state=42)
#
# elif METHOD == "umap":
#     reducer = umap.UMAP(n_components=3)
#
# else:
#     raise ValueError("Invalid METHOD. Choose from 'pca', 'tsne', 'umap'")
#
# # -------------------------------
# # DIMENSION REDUCTION
# # -------------------------------
# if METHOD == "tsne":
#     # t-SNE cannot transform new points → combine first
#     combined = np.vstack([doc_embeddings, query_embedding])
#     reduced_all = reducer.fit_transform(combined)
#
#     reduced_vectors = reduced_all[:-1]
#     query_reduced = reduced_all[-1:]
#
# else:
#     reduced_vectors = reducer.fit_transform(doc_embeddings)
#     query_reduced = reducer.transform(query_embedding)
#
# # -------------------------------
# # 3D PLOTTING
# # -------------------------------
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot documents
# for i, text in enumerate(documents):
#     x, y, z = reduced_vectors[i]
#     ax.scatter(x, y, z)
#     ax.text(x, y, z, text, fontsize=8)
#
# # Plot query
# qx, qy, qz = query_reduced[0]
# ax.scatter(qx, qy, qz, marker='x', s=120)
# ax.text(qx, qy, qz, "QUERY", fontsize=10)
#
# ax.set_title(f"3D Vector Space Visualization ({METHOD.upper()})")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
#
# plt.show()