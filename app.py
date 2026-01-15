import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(page_title="Embedding-Based Text Search", layout="centered")

st.title("🔍 Embedding-Based Text Search Engine")
st.write("Semantic search using sentence embeddings")

# Sample documents
documents = [
    "Natural Language Processing is a branch of Artificial Intelligence",
    "Machine learning is widely used in NLP applications",
    "Deep learning models improve language understanding",
    "Python is commonly used for data science and AI",
    "Text embeddings capture semantic meaning of sentences",
    "Cosine similarity is used to measure vector similarity",
    "Transformers are powerful models for NLP tasks"
]

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Generate embeddings
doc_embeddings = model.encode(documents)

# User input
query = st.text_input("Enter your search query:")

top_k = st.slider("Number of results", 1, 5, 3)

if query:
    query_embedding = model.encode([query])
    similarity_scores = cosine_similarity(query_embedding, doc_embeddings)[0]

    top_indices = np.argsort(similarity_scores)[-top_k:][::-1]

    st.subheader("🔎 Search Results:")
    for idx in top_indices:
        st.write(f"**Score:** {similarity_scores[idx]:.3f}")
        st.write(f"- {documents[idx]}")
        st.write("---")
