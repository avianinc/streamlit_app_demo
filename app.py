import streamlit as st
import os
import pandas as pd
import pdfplumber
import docx
import markdown
import json
from pathlib import Path
import requests

DATA_FOLDER = "data"

# Create data folder if it doesn't exist
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# Supported file types and their extensions
SUPPORTED_FILE_TYPES = {
    "PDF": ["pdf"],
    "Word": ["docx"],
    "Excel": ["xlsx"],
    "Text": ["txt"],
    "Markdown": ["md"],
    "CSV": ["csv"]
}

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
            with open(file_path, "r") as f:
                extracted_text = f.read()
        elif file_extension == ".md":
            with open(file_path, "r") as f:
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
    uploaded_files = st.file_uploader("Upload files", type=[ext for exts in SUPPORTED_FILE_TYPES.values() for ext in exts], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_uploaded_file(uploaded_file)
        st.success("Files uploaded and saved successfully!")

    # Process all files in data folder
    existing_files = list(Path(DATA_FOLDER).glob("*"))
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

        # Prepare the prompt
        all_text = "\n".join([extract_text_from_file(file) for file in existing_files])
        prompt = f"You are a helpful assistant. The following is the content of the uploaded files:\n\n{all_text}\n\nAnswer the following question:\n{user_input}"

        # Send the prompt to the Ollama server with streaming enabled
        url = "http://localhost:11434/api/generate"  # Using the endpoint that worked with curl
        headers = {"Content-Type": "application/json"}
        data = {
            "model": selected_model,  # Use the selected model
            "prompt": prompt
        }

        # Placeholder for streaming response
        response_placeholder = st.empty()
        full_response = ""

        try:
            with requests.post(url, json=data, headers=headers, stream=True) as response:
                if response.status_code == 200:
                    # Process each chunk as it comes in
                    for line in response.iter_lines():
                        if line:
                            try:
                                # Parse the incoming JSON line
                                parsed_obj = json.loads(line.decode('utf-8'))
                                response_chunk = parsed_obj.get("response", "")

                                # Accumulate and display the response
                                full_response += response_chunk
                                response_placeholder.chat_message("assistant").markdown(full_response)  # Update with Markdown formatting

                            except json.JSONDecodeError as e:
                                st.error(f"Failed to parse JSON part: {e}")
                                st.write("Problematic JSON part:", line)

                    # Update the chat history with the final response
                    st.session_state.history.append({"role": "assistant", "content": full_response})
                else:
                    st.error(f"Error communicating with Ollama server: {response.status_code}")
                    st.error(response.text)  # Print server error message to debug
        except requests.exceptions.RequestException as e:
            st.error(f"Error communicating with Ollama server: {e}")

if __name__ == "__main__":
    main()