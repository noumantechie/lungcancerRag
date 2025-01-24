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
client = Groq(api_key="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")

# Load the dataset
dataset_path = '/mount/src/lungcancerrag/app.py'  # Ensure this file is uploaded
df = pd.read_csv("dataset_path")

# Preview the dataset
st.write("Dataset preview:", df.head())

# Prepare embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # Open-source embedding model

# Convert dataset rows to embeddings
def row_to_text(row):
    return " ".join(f"{col}: {val}" for col, val in row.items())

df['text'] = df.apply(row_to_text, axis=1)
df['embedding'] = df['text'].apply(lambda x: model.encode(x))

# Define retrieval function
def retrieve_relevant_rows(query, top_n=3):
    query_embedding = model.encode(query)
    embeddings = np.vstack(df['embedding'].to_numpy())
    similarities = cosine_similarity([query_embedding], embeddings).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    return df.iloc[top_indices]

# RAG Functionality
def rag_pipeline(query):
    # Step 1: Retrieve relevant rows
    retrieved_rows = retrieve_relevant_rows(query, top_n=3)

    # Step 2: Combine retrieved data for the Groq model
    retrieved_text = " ".join(retrieved_rows['text'].tolist())
    input_to_groq = f"Context: {retrieved_text} \nQuestion: {query}"

    # Step 3: Use Groq for text generation
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an expert in analyzing medical data related to lung cancer.",
            },
            {
                "role": "user",
                "content": input_to_groq,
            }
        ],
        model="llama3-8b-8192",  # Use Groq's Llama model
    )
    return chat_completion.choices[0].message.content

# Streamlit interface
st.title("Medical Query Answering System")
st.write("Enter a query below and get a detailed response based on the dataset.")

# User input query
query = st.text_input("Your Query", "")

# Handle user input and show results
if query:
    response = rag_pipeline(query)
    st.write("Response:", response)
