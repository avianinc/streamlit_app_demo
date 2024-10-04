import streamlit as st
import os
import pandas as pd
import pdfplumber
import docx
import json
from pathlib import Path
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from sentence_transformers import SentenceTransformer
import uuid  # Import uuid module

DATA_FOLDER = "data"

# Create data folder if it doesn't exist
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# Initialize Qdrant client (assuming Qdrant is running locally)
qdrant_client = QdrantClient(host='localhost', port=6333)

# Define the collection name
COLLECTION_NAME = "document_chunks"

# Create a collection if it doesn't exist
try:
    qdrant_client.get_collection(collection_name=COLLECTION_NAME)
except Exception:
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qdrant_models.VectorParams(size=384, distance=qdrant_models.Distance.COSINE),
    )

# Supported file types and their extensions
SUPPORTED_FILE_TYPES = {
    "PDF": ["pdf"],
    "Word": ["docx"],
    "Excel": ["xlsx"],
    "Text": ["txt"],
    "Markdown": ["md"],
    "CSV": ["csv"]
}

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Helper function to save uploaded file
def save_uploaded_file(uploaded_file):
    with open(os.path.join(DATA_FOLDER, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

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
        elif file_extension == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                extracted_text = f.read()
        elif file_extension == ".md":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                extracted_text = f.read()
        elif file_extension == ".csv":
            df = pd.read_csv(file_path)
            extracted_text = df.to_string()
        elif file_extension == ".xlsx":
            df = pd.read_excel(file_path, None)  # Read all sheets
            for sheet_name, sheet_data in df.items():
                extracted_text += f"\nSheet: {sheet_name}\n"
                extracted_text += sheet_data.to_string()
    except Exception as e:
        extracted_text += f"\n[Error reading file {file_path.name}: {e}]"

    return extracted_text

# Function to chunk text
def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Function to process and store chunks in Qdrant
def process_and_store_chunks(file_path):
    text = extract_text_from_file(file_path)
    st.write(f"Extracted text from {file_path.name}: {text[:100]}...")
    chunks = chunk_text(text)
    st.write(f"Number of chunks created: {len(chunks)}")
    embeddings = embedding_model.encode(chunks)
    st.write(f"Embeddings shape: {embeddings.shape}")
    # Ensure embeddings have the expected dimension
    if len(embeddings.shape) == 1 or embeddings.shape[1] != 384:
        st.error(f"Unexpected embedding dimension: {embeddings.shape}")
    payloads = [{'text': chunk, 'file_name': file_path.name} for chunk in chunks]
    vectors = embeddings.tolist()
    # Generate UUIDs for each vector
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
        st.write(f"Successfully upserted {len(points)} points into Qdrant.")
        count_result = qdrant_client.count(collection_name=COLLECTION_NAME)
        st.write(f"Total points in Qdrant: {count_result.count}")
    except Exception as e:
        st.error(f"Error during upsert: {e}")

# Function to retrieve relevant chunks from Qdrant
def get_relevant_chunks(query, top_k=5):
    query_embedding = embedding_model.encode([query])[0]
    try:
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            limit=top_k,
        )
        st.write(f"Search result: {search_result}")
        # Check if search_result is empty
        if not search_result:
            st.write("No relevant chunks found in Qdrant.")
        # Extract the texts of the top matching chunks
        relevant_chunks = [hit.payload['text'] for hit in search_result]
        return relevant_chunks
    except Exception as e:
        st.error(f"Error during search: {e}")
        return []

# Streamlit UI
def main():
    st.title("Multi-File Chat with Ollama")

    # Fetch list of models from the server
    model_url = "http://localhost:11434/v1/models"
    try:
        response = requests.get(model_url)
        if response.status_code == 200:
            models_data = response.json()
            model_names = [model['id'] for model in models_data.get("data", [])]
        else:
            st.error(f"Error fetching models: {response.status_code}")
            model_names = ["mistral"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with Ollama server: {e}")
        model_names = ["mistral"]

    # Model selector
    selected_model = st.selectbox("Select a model to use", model_names)

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload files",
        type=[ext for exts in SUPPORTED_FILE_TYPES.values() for ext in exts],
        accept_multiple_files=True
    )

    # Handle file uploads
    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_uploaded_file(uploaded_file)
            file_path = Path(DATA_FOLDER) / uploaded_file.name
            process_and_store_chunks(file_path)
        st.success("Files uploaded, processed, and stored successfully!")
        # No need to rerun; the UI will update automatically

    # Refresh the list of existing files
    existing_files = list(Path(DATA_FOLDER).glob("*"))

    # Manage existing files in the data folder using a sidebar combo box
    if existing_files:
        st.sidebar.write("### Manage Existing Files")
        selected_file = st.sidebar.selectbox(
            "Select a file to delete", [file.name for file in existing_files]
        )
        if st.sidebar.button("Delete Selected File"):
            file_to_delete = Path(DATA_FOLDER) / selected_file
            if file_to_delete.exists():
                os.remove(file_to_delete)
                st.sidebar.success(f"Deleted {selected_file}")
                # Delete associated vectors from Qdrant using a Filter
                try:
                    qdrant_client.delete(
                        collection_name=COLLECTION_NAME,
                        # Use the `filter` parameter with `qdrant_models.Filter`
                        filter=qdrant_models.Filter(
                            must=[
                                qdrant_models.FieldCondition(
                                    key="file_name",
                                    match=qdrant_models.MatchValue(value=selected_file)
                                )
                            ]
                        )
                    )
                    st.write(f"Deleted vectors associated with {selected_file} from Qdrant.")
                except Exception as e:
                    st.error(f"Error deleting vectors from Qdrant: {e}")
                # Update the list of existing files after deletion
                existing_files = list(Path(DATA_FOLDER).glob("*"))

    # Process all files in data folder
    if existing_files:
        with st.expander("Extracted Content from Files"):
            for file in existing_files:
                st.subheader(file.name)
                extracted_text = extract_text_from_file(file)
                st.text_area(f"Extracted Content from {file.name}", extracted_text, height=200)

    # Chat interface using Streamlit's chat elements
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Display chat history
    for chat in st.session_state.history:
        if chat['role'] == 'user':
            st.chat_message("user").markdown(chat['content'])
        elif chat['role'] == 'assistant':
            st.chat_message("assistant").markdown(chat['content'])

    # User input for chat
    user_input = st.chat_input("Ask a question about the files")
    if user_input:
        # Add user's question to history
        st.session_state.history.append({"role": "user", "content": user_input})
        st.chat_message("user").markdown(user_input)

        # Retrieve relevant chunks
        relevant_chunks = get_relevant_chunks(user_input, top_k=5)
        context = "\n".join(relevant_chunks)

        # Prepare the prompt
        prompt = f"You are a helpful assistant. Use the following context to answer the question:\n\n{context}\n\nQuestion:\n{user_input}"

        # Log the prompt for debugging
        st.write("Prompt being sent to the assistant:")
        st.code(prompt)

        # Send the prompt to the Ollama server with streaming enabled
        url = "http://localhost:11434/api/generate"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": selected_model,
            "prompt": prompt
        }

        try:
            with requests.post(url, json=data, headers=headers, stream=True) as response:
                if response.status_code == 200:
                    # Create a container for the assistant's response
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        response_text = ""
                        # Process each chunk as it comes in
                        for line in response.iter_lines():
                            if line:
                                try:
                                    # Parse the incoming JSON line
                                    parsed_obj = json.loads(line.decode('utf-8'))
                                    response_chunk = parsed_obj.get("response", "")
                                    # Accumulate and display the response
                                    response_text += response_chunk
                                    message_placeholder.markdown(response_text)
                                except json.JSONDecodeError as e:
                                    st.error(f"Failed to parse JSON part: {e}")
                                    st.write("Problematic JSON part:", line)
                    # Update the chat history with the final response
                    st.session_state.history.append({"role": "assistant", "content": response_text})
                else:
                    st.error(f"Error communicating with Ollama server: {response.status_code}")
                    st.error(response.text)
        except requests.exceptions.RequestException as e:
            st.error(f"Error communicating with Ollama server: {e}")

if __name__ == "__main__":
    main()
