import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Information Retrieval App",
    page_icon="🔍",
    layout="centered"
)

# -----------------------------
# Load documents
# -----------------------------
@st.cache_data
def load_documents():
    with open("documents.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

documents = load_documents()

# -----------------------------
# Create TF-IDF embeddings
# -----------------------------
@st.cache_data
def create_embeddings(docs):
    vectorizer = TfidfVectorizer(stop_words="english")
    embeddings = vectorizer.fit_transform(docs).toarray()
    return vectorizer, embeddings

vectorizer, embeddings = create_embeddings(documents)

# -----------------------------
# Query embedding function
# -----------------------------
def get_query_embedding(query):
    return vectorizer.transform([query]).toarray()[0]

# -----------------------------
# Retrieval function
# -----------------------------
def retrieve_top_k(query_embedding, embeddings, k=5):
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        embeddings
    )[0]

    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(documents[i], similarities[i]) for i in top_k_indices]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🔍 Information Retrieval using Document Embeddings")

st.markdown(
    """
    **Student:** Lynn Younes  
    **Student ID:** 60107070  
    **Course:** DSAI4201 – Selected Topics in Data Science & AI
    """
)

st.write(
    "This application retrieves the most relevant documents using "
    "**TF-IDF document embeddings** and **cosine similarity**."
)

query = st.text_input("Enter your query:")
k = st.slider("Number of documents to retrieve (K)", min_value=1, max_value=10, value=5)

if st.button("Search"):
    if query.strip() == "":
        st.warning("Please enter a query.")
    else:
        with st.spinner("Searching for relevant documents..."):
            query_embedding = get_query_embedding(query)
            results = retrieve_top_k(query_embedding, embeddings, k)

        st.subheader(f"Top {k} Relevant Documents")
        for i, (doc, score) in enumerate(results, start=1):
            st.markdown(f"**{i}. {doc}**")
            st.caption(f"Cosine similarity score: {score:.4f}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "Developed by Lynn Younes (60107070) | "
    "DSAI4201 Information Retrieval Lab | "
    "Built with Streamlit, TF-IDF, and cosine similarity"
)