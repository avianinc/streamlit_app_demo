import streamlit as st
import os
import shutil
import pandas as pd
import pdfplumber
import docx
from pathlib import Path
from openai import OpenAI
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from sentence_transformers import SentenceTransformer
import uuid
import logging
import json
from utilities.icon import page_icon

# Initialize Streamlit
st.set_page_config(
    page_title="JS5000",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Configuration Variables ---------------- #

# Data folder for uploaded files
DATA_FOLDER = "data"

# Qdrant settings
QDRANT_HOST = 'qdrant'
QDRANT_PORT = 6333
COLLECTION_NAME = "document_chunks"

# Embedding model
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# Text chunking settings
CHUNK_SIZE = 300
OVERLAP = 100

# Supported file types and their extensions
SUPPORTED_FILE_TYPES = {
    "PDF": ["pdf"],
    "Word": ["docx"],
    "Excel": ["xlsx"],
    "Text": ["txt"],
    "Markdown": ["md"],
    "CSV": ["csv"]
}

# Logging Configuration
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ---------------- Initialize Ollama Client ---------------- #

# Initialize the client for Ollama server
client = OpenAI(
    base_url='http://ollama:11434/v1/',  # Updated base URL for Docker environment
    api_key='ollama',  # Required, but unused
)

# ---------------- Initialize Qdrant Client ---------------- #

# Initialize Qdrant client
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Ensure the Qdrant collection exists
def ensure_qdrant_collection():
    try:
        qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        logging.info("Qdrant collection exists.")
    except Exception:
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qdrant_models.VectorParams(size=384, distance=qdrant_models.Distance.COSINE),
        )
        logging.info("Qdrant collection created.")

# ---------------- Helper Functions ---------------- #

# Initialize the embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Helper function to save uploaded file
def save_uploaded_file(uploaded_file):
    os.makedirs(DATA_FOLDER, exist_ok=True)
    file_path = os.path.join(DATA_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    logging.info(f"Saved uploaded file: {file_path}")
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
            df_dict = pd.read_excel(file_path, sheet_name=None)
            for sheet_name, sheet_data in df_dict.items():
                extracted_text += f"\nSheet: {sheet_name}\n"
                extracted_text += sheet_data.to_string()
    except Exception as e:
        st.error(f"Error reading file {file_path.name}: {e}")
        logging.error(f"Error reading file {file_path.name}: {e}")

    logging.info(f"Extracted text from {file_path.name}")
    return extracted_text

# Function to chunk text
def chunk_text(text):
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - OVERLAP):
        chunk = ' '.join(words[i:i + CHUNK_SIZE])
        chunks.append(chunk)
    logging.info(f"Created {len(chunks)} chunks.")
    return chunks

# Function to process and store chunks in Qdrant
def process_and_store_chunks(file_path):
    text = extract_text_from_file(file_path)
    chunks = chunk_text(text)
    if not chunks:
        st.warning(f"No text extracted from {Path(file_path).name}.")
        logging.warning(f"No text extracted from {Path(file_path).name}.")
        return

    try:
        embeddings = embedding_model.encode(chunks)
        logging.info("Generated embeddings.")
    except Exception as e:
        st.error(f"Error during embedding: {e}")
        logging.error(f"Error during embedding: {e}")
        return

    if len(embeddings.shape) == 1 or embeddings.shape[1] != 384:
        st.error(f"Unexpected embedding dimension: {embeddings.shape}")
        logging.error(f"Unexpected embedding dimension: {embeddings.shape}")
        return

    payloads = [{'text': chunk, 'file_name': Path(file_path).name} for chunk in chunks]
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
        st.success(f"Upserted {len(chunks)} chunks to Qdrant.")
        logging.info(f"Upserted {len(chunks)} chunks to Qdrant.")
    except Exception as e:
        st.error(f"Error during upsert to Qdrant: {e}")
        logging.error(f"Error during upsert to Qdrant: {e}")

# Function to retrieve relevant chunks from Qdrant
def get_relevant_chunks(query, top_k=5):
    try:
        query_embedding = embedding_model.encode([query])[0]
        logging.info("Generated query embedding.")
    except Exception as e:
        st.error(f"Error during query embedding: {e}")
        logging.error(f"Error during query embedding: {e}")
        return []

    try:
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            limit=top_k,
        )
        if not search_result:
            st.info("No relevant chunks found.")
            logging.info("No relevant chunks found.")
            return []
        relevant_chunks = [hit.payload['text'] for hit in search_result]
        logging.info(f"Retrieved {len(relevant_chunks)} relevant chunks.")
        return relevant_chunks
    except Exception as e:
        st.error(f"Error during search in Qdrant: {e}")
        logging.error(f"Error during search in Qdrant: {e}")
        return []

# Function to download a model from Ollama with progress feedback
def download_model(model_name):
    try:
        url = "http://ollama:11434/api/pull"
        headers = {"Content-Type": "application/json"}
        data = {"name": model_name, "stream": True}

        # Send POST request with streaming enabled
        with requests.post(url, json=data, headers=headers, stream=True) as response:
            response.raise_for_status()

            st.info(f"Downloading model '{model_name}'...")
            logging.info(f"Started downloading model '{model_name}'.")

            status_placeholder = st.empty()  # Create a placeholder for the status

            # Process the streaming response
            for chunk in response.iter_lines():
                if chunk:
                    chunk_data = json.loads(chunk.decode('utf-8'))
                    status = chunk_data.get("status", "unknown status")

                    if "completed" in chunk_data and "total" in chunk_data:
                        completed = chunk_data["completed"]
                        total = chunk_data["total"]
                        percent_complete = (completed / total) * 100

                        # Update status with percentage
                        status_placeholder.write(
                            f"Status: {status} - {completed}/{total} bytes downloaded ({percent_complete:.2f}% complete)"
                        )
                    else:
                        status_placeholder.write(f"Status: {status}")

                    # Log the status
                    logging.info(f"Download status: {status}")

            st.success(f"Model '{model_name}' downloaded successfully!")
            logging.info(f"Model '{model_name}' downloaded successfully!")
    except requests.exceptions.RequestException as e:
        st.error(f"Error occurred: {e}")
        logging.error(f"Error occurred while downloading model '{model_name}': {e}")

# Streamlit UI
def main():
    st.title("JARED5000")

    # Ensure the Qdrant collection exists
    ensure_qdrant_collection()

    # Fetch list of models from the Ollama client
    try:
        models_info = client.models.list()
        # Exclude LLava models from the model list
        model_names = [model.id for model in models_info.data if 'llava' not in model.id.lower()]
        logging.info("Fetched models from Ollama server.")
    except Exception as e:
        st.error(f"Error fetching models from Ollama server: {e}")
        logging.error(f"Error fetching models from Ollama server: {e}")
        model_names = []  # Empty list signifies no models available

    if not model_names:
        st.warning("No models available on the Ollama server. Please download a model using the sidebar.")
        logging.warning("No models available on the Ollama server.")
    else:
        # Set default model
        default_model = "llama3.2:1b"
        if default_model not in model_names:
            default_model = model_names[0]  # Fallback to the first model if default is not available

        # Model selector
        selected_model = st.selectbox("Select a model to use", model_names, index=model_names.index(default_model))

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
            logging.info("Cleared data folder.")
        os.makedirs(DATA_FOLDER, exist_ok=True)
        ensure_qdrant_collection()
        st.success("Cleared existing data.")
        logging.info("Cleared existing data.")

    # Handle file uploads
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = Path(save_uploaded_file(uploaded_file))
            process_and_store_chunks(file_path)
        st.success("Files uploaded and processed successfully!")
        logging.info("Uploaded and processed files.")

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
                logging.info(f"Deleted file: {file_to_delete}")
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
                    logging.info(f"Deleted vectors for file: {selected_file}")
                except Exception as e:
                    st.sidebar.error(f"Error deleting data from Qdrant: {e}")
                    logging.error(f"Error deleting data from Qdrant: {e}")

    # Checkbox to display extracted content
    display_content = st.checkbox("Display extracted content from files", value=False)

    # Display extracted content
    if display_content and existing_files:
        with st.expander("View Extracted Content from Files"):
            for file in existing_files:
                extracted_text = extract_text_from_file(file)
                if extracted_text:
                    st.subheader(file.name)
                    st.text_area("", extracted_text, height=150)

    # Download Tool in Sidebar
    st.sidebar.header("Download Ollama Model")
    with st.sidebar.form("download_form"):
        model_to_download = st.text_input("Enter model name from Ollama Library")
        submit_button = st.form_submit_button("Download Model")
        if submit_button:
            if model_to_download.strip():
                download_model(model_to_download.strip())
            else:
                st.sidebar.error("Please enter a valid model name.")
                logging.warning("Empty model name entered for download.")

    # Chat interface
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Display chat history
    for chat in st.session_state.history:
        with st.chat_message(chat['role']):
            st.markdown(chat['content'])

    # User input for chat
    user_input = st.chat_input("Ask a question about the files")

    if user_input and model_names:
        # Display user's message
        with st.chat_message("user"):
            st.markdown(user_input)
        # Append user's message to the chat history
        st.session_state.history.append({"role": "user", "content": user_input})
        logging.info(f"User input: {user_input}")

        # Retrieve relevant chunks
        relevant_chunks = get_relevant_chunks(user_input, top_k=5)
        context = "\n".join(relevant_chunks)

        # Build conversation history
        conversation = ""
        for chat in st.session_state.history:
            role = chat['role']
            content = chat['content']
            conversation += f"{role.capitalize()}: {content}\n"

        # Prepare the prompt with conversation history
        prompt = (
            f"You are a helpful assistant. Continue the following conversation using the provided context. "
            f"Do not mention the context or search results in your answer. Think through the answer internally, "
            f"and provide a concise and clear response to the user.\n\n"
            f"Context:\n{context}\n\n"
            f"Conversation:\n{conversation}\n"
            f"Assistant:"
        )

        # Send the prompt to the Ollama server using OpenAI client
        try:
            # Display the assistant's response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                response_text = ""

                # Send request with streaming
                response = client.completions.create(
                    model=selected_model,
                    prompt=prompt,
                    stream=True
                )

                # Process the stream response as it's received
                for chunk in response:
                    response_chunk = chunk.choices[0].text
                    response_text += response_chunk
                    message_placeholder.markdown(response_text)

            st.session_state.history.append({"role": "assistant", "content": response_text})
            logging.info(f"Assistant response: {response_text}")
        except Exception as e:
            st.error(f"Error occurred while generating response: {e}")
            logging.error(f"Error occurred while generating response: {e}")

    elif user_input and not model_names:
        st.error("No models available on the Ollama server. Please download a model using the sidebar.")
        logging.warning("User attempted to chat without available models.")

    # Sidebar links
    with st.sidebar:
        st.markdown("## Useful Links")
        st.markdown("[Qdrant Dashboard](http://localhost:6333/dashboard)")
        st.markdown("[Ollama Library](https://ollama.com/library)")

if __name__ == "__main__":
    # Create data folder if it doesn't exist
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
    main()
