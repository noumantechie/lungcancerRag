# Import libraries
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from groq import Groq
import streamlit as st
import csv

# Set up Groq API
try:
    secrets = st.secrets["groq_api_key"]
except KeyError:
    st.error("API Key not found in secrets. Please check the Streamlit Secrets configuration.")

# Set up Groq API with the secret key
client = Groq(api_key=secrets)

# Load the dataset
dataset_path = 'https://raw.githubusercontent.com/noumantechie/RagApplication/main/lungcaner/dataseter.csv'  # Ensure this file is uploaded
df = pd.read_csv(dataset_path)

# Prepare embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # Open-source embedding model

# Convert dataset rows to embeddings
def row_to_text(row):
    return " ".join(f"{col}: {val}" for col, val in row.items())

df['text'] = df.apply(row_to_text, axis=1)
embeddings = np.vstack(df['text'].apply(lambda x: model.encode(x)).to_numpy())

# Define retrieval function
def retrieve_relevant_rows(query, top_n=3):
    query_embedding = model.encode(query)
    similarities = cosine_similarity([query_embedding], embeddings).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    return df.iloc[top_indices]

# In-memory cache for query-response pairs
query_cache = {}

# RAG Functionality
def rag_pipeline(query):
    # Check if the query is already in the cache
    if query in query_cache:
        return query_cache[query]
    
    # Step 1: Retrieve relevant rows
    retrieved_rows = retrieve_relevant_rows(query, top_n=3)

    # Step 2: Combine retrieved data for the Groq model
    retrieved_text = " ".join(retrieved_rows['text'].tolist())
    input_to_groq = f"Context: {retrieved_text} \nQuestion: {query}"

    # Step 3: Use Groq for text generation
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "system",
            "content": "You are an expert in analyzing medical data related to lung cancer.",
        },
        {
            "role": "user",
            "content": input_to_groq,
        }],
        model="llama3-8b-8192",  # Use Groq's Llama model
    )

    # Get the response
    response = chat_completion.choices[0].message.content

    # Store the query and response in the cache
    query_cache[query] = response

    return response

# Streamlit interface
st.title("Medical Query Answering System")
st.write("Enter a query below and get a detailed response based on the dataset.")

# User input query
query = st.text_input("Your Query", "")

# Handle user input and show results
if query:
    response = rag_pipeline(query)
    st.write("Response:", response)
