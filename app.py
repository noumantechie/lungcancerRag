import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from groq import Groq
import streamlit as st
import csv
from tqdm import tqdm
import time

# Set up Groq API
if "groq_api_key" in st.secrets:
    secrets = st.secrets["groq_api_key"]
else:
    st.error("API Key not found in secrets. Please check the Streamlit Secrets configuration.")
    st.stop()

# Initialize Groq client
client = Groq(api_key=secrets)

# Load the dataset
dataset_path = 'https://raw.githubusercontent.com/noumantechie/lungcancerRag/main/dataseter.csv'
try:
    df = pd.read_csv(dataset_path)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.write("Please check if the dataset URL is correct or reachable.")
    st.stop()

# Prepare embeddings (caching embeddings to avoid recomputation)
model = SentenceTransformer('all-MiniLM-L6-v2')  # Open-source embedding model

@st.cache_data
def compute_embeddings(df, exclude_columns=[]):
    def row_to_text(row):
        return " ".join(f"{col}: {val}" for col, val in row.items() if col not in exclude_columns)

    df['text'] = df.apply(row_to_text, axis=1)
    texts = df['text'].tolist()
    embeddings = np.array([model.encode(text) for text in tqdm(texts, desc="Generating Embeddings")])
    return embeddings

embeddings = compute_embeddings(df)

# Define retrieval function
def retrieve_relevant_rows(query, top_n=3):
    if embeddings is None:
        st.error("Embeddings not computed. Please check the data and embeddings.")
        return None
    query_embedding = model.encode(query)
    similarities = cosine_similarity([query_embedding], embeddings).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    return df.iloc[top_indices]

# Call Groq API with retry mechanism
def call_groq_api(input_to_groq, retries=3):
    for attempt in range(retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[{
                    "role": "system",
                    "content": "You are an expert in analyzing medical data related to lung cancer. Reject any unrelated queries and inform the user.",
                },
                {
                    "role": "user",
                    "content": input_to_groq,
                }],
                model="llama3-8b-8192",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                st.error(f"Groq API call failed after {retries} attempts: {e}")
                return None

# RAG Functionality
def rag_pipeline(query):
    retrieved_rows = retrieve_relevant_rows(query, top_n=3)

    if retrieved_rows is None or retrieved_rows.empty:
        return "No relevant data found for the query. Please ensure your query is related to lung cancer or the dataset."

    retrieved_text = " ".join(retrieved_rows['text'].tolist())
    input_to_groq = f"Context: {retrieved_text} \nQuestion: {query}"

    response_content = call_groq_api(input_to_groq)
    if response_content:
        return response_content
    else:
        return "There was an issue processing your request."

# Streamlit interface
st.title("Medical Query Answering System")
st.write("Enter a query below to get a detailed response based on the dataset.")

# User input query with placeholder text
query = st.text_input("Your Query", placeholder="E.g., What are the symptoms of lung cancer?")

# Add a button to execute the query
if st.button("Get Response"):
    if query.strip():
        if len(query.strip()) > 3:
            with st.spinner('Generating response...'):
                response = rag_pipeline(query)
            st.write("### Response")
            st.write(response)
        else:
            st.warning("Please enter a longer query.")
    else:
        st.warning("The query field is empty. Please enter a query to proceed.")

# Style the interface
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)
