# Information Retrieval App using Document Embeddings

This project is a simple Information Retrieval (IR) web application built with **Streamlit**.  
It retrieves the top-K most relevant documents for a user query using **TF-IDF document embeddings** and **cosine similarity**.

## Features
- TF-IDF–based document and query embeddings
- Cosine similarity for ranking documents
- Interactive Top-K document retrieval
- Deployed using Streamlit Cloud

## Technologies Used
- Python
- Streamlit
- NumPy
- Scikit-learn (TF-IDF, cosine similarity)

## How It Works
1. Documents are converted into TF-IDF embeddings.
2. The user query is embedded into the same vector space.
3. Cosine similarity is computed between the query and all documents.
4. The top-K most similar documents are displayed.

## Author
**Lynn Younes**  
Student ID: **60107070**  
Course: **DSAI4201 – Selected Topics in Data Science & AI**
