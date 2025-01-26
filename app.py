# Import libraries
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from groq import Groq
import streamlit as st

# Set up Groq API
try:
    api_key = st.secrets["groq_api_key"]
except KeyError:
    st.error("API Key not found in secrets. Please check the Streamlit Secrets configuration.")
    st.stop()

# Initialize the Groq API client
client = Groq(api_key=api_key)

# Load the dataset (ensure the file exists at the given path or provide an online link)
dataset_path = 'https://raw.githubusercontent.com/noumantechie/RagApplication/main/lungcaner/dataseter.csv'
df = pd.read_csv(dataset_path)

# Preprocess the dataset
df.fillna("", inplace=True)  # Fill missing values
df['combined'] = df.astype(str).apply(' '.join, axis=1)  # Combine all columns into a single string

# Initialize the SentenceTransformer model once
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the dataset (only once)
embeddings = model.encode(df['combined'].tolist())
vector_database = np.array(embeddings)

# Define a function to retrieve the most similar rows
def retrieve_relevant_rows(query, top_n=3):
    query_embedding = model.encode([query])  # Generate embedding for the query
    similarities = cosine_similarity(query_embedding, vector_database)  # Compute similarities
    top_indices = np.argsort(similarities[0])[::-1][:top_n]  # Get top_n indices
    return df.iloc[top_indices], top_indices

# Define the RAG pipeline
def rag_pipeline(query):
    # Retrieve relevant rows
    retrieved_rows, _ = retrieve_relevant_rows(query, top_n=3)

    # Combine retrieved data into a single context
    retrieved_text = " ".join(retrieved_rows['combined'].tolist())
    input_to_groq = f"Context: {retrieved_text} \nQuestion: {query}"

    # Generate a response using the Groq API
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an expert in analyzing medical data related to lung cancer."},
            {"role": "user", "content": input_to_groq},
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

# Streamlit interface
st.title("Lung Cancer Predictor App")

# User input query
query = st.text_input("Enter your query:")

# Handle user input and display results
if st.button("Get Prediction") and query:
    with st.spinner("Processing your query..."):
        response = rag_pipeline(query)
    st.write("Response:", response)
