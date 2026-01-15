import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(
    page_title="Embedding-Based Text Search Engine",
    page_icon="🔍",
    layout="centered"
)

# Title & description
st.title("🔍 Embedding-Based Text Search Engine")
st.markdown(
    """
    **Semantic search using sentence embeddings**  
    This app retrieves *meaningfully similar* sentences using transformer-based embeddings  
    instead of simple keyword matching.
    """
)

st.divider()

# Example queries
with st.expander("💡 Example search queries"):
    st.markdown("""
    - machine learning in NLP  
    - artificial intelligence for language processing  
    - python for data science and AI  
    - how sentence embeddings work  
    """)

# Sample documents (knowledge base)
documents = [
    "Natural Language Processing is a branch of Artificial Intelligence",
    "Machine learning is widely used in NLP applications",
    "Deep learning models improve language understanding",
    "Python is commonly used for data science and AI",
    "Text embeddings capture semantic meaning of sentences",
    "Cosine similarity measures similarity between vectors",
    "Transformer models are powerful for NLP tasks"
]

# Load model (cached)
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()
doc_embeddings = model.encode(documents)

# User input
query = st.text_input("🔎 Enter your search query:")

top_k = st.slider(
    "Number of results to display",
    min_value=1,
    max_value=5,
    value=3
)

# Search logic
if query.strip():
    query_embedding = model.encode([query])
    similarity_scores = cosine_similarity(query_embedding, doc_embeddings)[0]

    top_indices = np.argsort(similarity_scores)[-top_k:][::-1]

    st.divider()
    st.subheader("📊 Search Results")

    for rank, idx in enumerate(top_indices, start=1):
        st.markdown(
            f"""
            **{rank}. {documents[idx]}**  
            Similarity Score: `{similarity_scores[idx]:.3f}`
            """
        )
        st.write("---")
else:
    st.info("👆 Enter a meaningful sentence to perform semantic search.")
