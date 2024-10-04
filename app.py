import streamlit as st
import os
import shutil  # For deleting the data folder
import pandas as pd
import pdfplumber
import docx
import json
from pathlib import Path
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from sentence_transformers import SentenceTransformer
import uuid

# ---------------- Configuration Variables ---------------- #

# Data folder for uploaded files
DATA_FOLDER = "data"

# Qdrant settings
QDRANT_HOST = 'qdrant'  # Use 'qdrant' when running in Docker Compose, 'localhost' otherwise
QDRANT_PORT = 6333
COLLECTION_NAME = "document_chunks"

# Ollama settings
OLLAMA_HOST = 'ollama'  # Use 'ollama' when running in Docker Compose, 'localhost' otherwise
OLLAMA_PORT = 11434
MODEL_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/v1/models"
GENERATE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"

# Embedding model
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# Text chunking settings
CHUNK_SIZE = 300  # Number of words per chunk
OVERLAP = 100     # Number of words to overlap between chunks

# Supported file types and their extensions
SUPPORTED_FILE_TYPES = {
    "PDF": ["pdf"],
    "Word": ["docx"],
    "Excel": ["xlsx"],
    "Text": ["txt"],
    "Markdown": ["md"],
    "CSV": ["csv"]
}

# --------------------------------------------------------- #

# Initialize Qdrant client
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Ensure the Qdrant collection exists
def ensure_qdrant_collection():
    try:
        qdrant_client.get_collection(collection_name=COLLECTION_NAME)
    except Exception:
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qdrant_models.VectorParams(size=384, distance=qdrant_models.Distance.COSINE),
        )

# Initialize the embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Helper function to save uploaded file
def save_uploaded_file(uploaded_file):
    file_path = os.path.join(DATA_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Helper function to extract text from various file types
def extract_text_from_file(file_path):
    file_extension = file_path.suffix.lower()
    extracted_text = ""

    try:
        if file_extension == ".pdf":
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    extracted_text += page.extract_text() or ""
        elif file_extension == ".docx":
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                extracted_text += para.text + "\n"
        elif file_extension in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                extracted_text = f.read()
        elif file_extension == ".csv":
            df = pd.read_csv(file_path)
            extracted_text = df.to_string()
        elif file_extension == ".xlsx":
            df_dict = pd.read_excel(file_path, sheet_name=None)  # Read all sheets
            for sheet_name, sheet_data in df_dict.items():
                extracted_text += f"\nSheet: {sheet_name}\n"
                extracted_text += sheet_data.to_string()
    except Exception as e:
        st.error(f"Error reading file {file_path.name}: {e}")

    return extracted_text

# Function to chunk text
def chunk_text(text):
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - OVERLAP):
        chunk = ' '.join(words[i:i + CHUNK_SIZE])
        chunks.append(chunk)
    return chunks

# Function to process and store chunks in Qdrant
def process_and_store_chunks(file_path):
    text = extract_text_from_file(file_path)
    chunks = chunk_text(text)
    if not chunks:
        st.warning(f"No text extracted from {file_path.name}.")
        return

    try:
        embeddings = embedding_model.encode(chunks)
    except Exception as e:
        st.error(f"Error during embedding: {e}")
        return

    if len(embeddings.shape) == 1 or embeddings.shape[1] != 384:
        st.error(f"Unexpected embedding dimension: {embeddings.shape}")
        return

    payloads = [{'text': chunk, 'file_name': file_path.name} for chunk in chunks]
    vectors = embeddings.tolist()
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

    points = [
        qdrant_models.PointStruct(
            id=ids[i],
            vector=vectors[i],
            payload=payloads[i]
        )
        for i in range(len(chunks))
    ]

    try:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
    except Exception as e:
        st.error(f"Error during upsert to Qdrant: {e}")

# Function to retrieve relevant chunks from Qdrant
def get_relevant_chunks(query, top_k=5):
    try:
        query_embedding = embedding_model.encode([query])[0]
    except Exception as e:
        st.error(f"Error during query embedding: {e}")
        return []

    try:
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            limit=top_k,
        )
        if not search_result:
            st.info("No relevant chunks found.")
        relevant_chunks = [hit.payload['text'] for hit in search_result]
        return relevant_chunks
    except Exception as e:
        st.error(f"Error during search in Qdrant: {e}")
        return []

def main():
    st.title("Multi-File Chat with Ollama")

    # Ensure the Qdrant collection exists
    ensure_qdrant_collection()

    # Fetch list of models from the server
    try:
        response = requests.get(MODEL_URL)
        response.raise_for_status()
        models_data = response.json()
        model_names = [model['id'] for model in models_data.get("data", [])]
    except Exception as e:
        st.error(f"Error fetching models from Ollama server: {e}")
        model_names = ["mistral"]  # Fallback model

    # Add the model selector here
    selected_model = st.selectbox("Select a model to use", model_names)

    # Option to clear existing data
    clear_data = st.checkbox("Clear existing data before uploading new files")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload files",
        type=[ext for exts in SUPPORTED_FILE_TYPES.values() for ext in exts],
        accept_multiple_files=True
    )

    if clear_data:
        # Clear the data folder and Qdrant collection
        if os.path.exists(DATA_FOLDER):
            shutil.rmtree(DATA_FOLDER)
        os.makedirs(DATA_FOLDER, exist_ok=True)
        ensure_qdrant_collection()
        st.success("Cleared existing data.")

    # Handle file uploads
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = Path(save_uploaded_file(uploaded_file))
            process_and_store_chunks(file_path)
        st.success("Files uploaded and processed successfully!")

    # Get the list of existing files
    existing_files = list(Path(DATA_FOLDER).glob("*"))

    # Manage existing files in the data folder
    if existing_files:
        st.sidebar.header("Manage Existing Files")
        selected_file = st.sidebar.selectbox(
            "Select a file to delete", [file.name for file in existing_files]
        )
        if st.sidebar.button("Delete Selected File"):
            file_to_delete = Path(DATA_FOLDER) / selected_file
            if file_to_delete.exists():
                os.remove(file_to_delete)
                # Delete associated vectors from Qdrant
                try:
                    qdrant_client.delete(
                        collection_name=COLLECTION_NAME,
                        points_selector=qdrant_models.Filter(
                            must=[
                                qdrant_models.FieldCondition(
                                    key="file_name",
                                    match=qdrant_models.MatchValue(value=selected_file)
                                )
                            ]
                        )
                    )
                    st.sidebar.success(f"Deleted {selected_file} and associated data.")
                except Exception as e:
                    st.sidebar.error(f"Error deleting data from Qdrant: {e}")

    # Display extracted content
    if existing_files:
        with st.expander("View Extracted Content from Files"):
            for file in existing_files:
                extracted_text = extract_text_from_file(file)
                if extracted_text:
                    st.subheader(file.name)
                    st.text_area("", extracted_text, height=150)

    # Chat interface
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Display chat history
    for chat in st.session_state.history:
        st.chat_message(chat['role']).markdown(chat['content'])

    # User input for chat
    user_input = st.chat_input("Ask a question about the files")
    if user_input:
        st.session_state.history.append({"role": "user", "content": user_input})

        # Retrieve relevant chunks
        relevant_chunks = get_relevant_chunks(user_input, top_k=5)
        context = "\n".join(relevant_chunks)

        # Prepare the prompt
        prompt = (
            f"You are a helpful assistant. Answer the following question using the provided context. "
            f"Do not mention the context or search results in your answer. Think through the answer internally, "
            f"and provide a concise and clear response to the user.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{user_input}"
        )

        # Send the prompt to the Ollama server
        headers = {"Content-Type": "application/json"}
        data = {
            "model": selected_model,
            "prompt": prompt
        }

        try:
            with requests.post(GENERATE_URL, json=data, headers=headers, stream=True) as response:
                response.raise_for_status()
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    response_text = ""
                    for line in response.iter_lines():
                        if line:
                            parsed_obj = json.loads(line.decode('utf-8'))
                            response_chunk = parsed_obj.get("response", "")
                            response_text += response_chunk
                            message_placeholder.markdown(response_text)
                st.session_state.history.append({"role": "assistant", "content": response_text})
        except Exception as e:
            st.error(f"Error communicating with Ollama server: {e}")

    # Sidebar links
    with st.sidebar:
        st.markdown("## Useful Links")
        st.markdown("[Qdrant Dashboard](http://localhost:6333/dashboard)")

if __name__ == "__main__":
    main()
