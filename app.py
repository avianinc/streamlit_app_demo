import streamlit as st
import os
import pandas as pd
import pdfplumber
import docx
import markdown
import json
from pathlib import Path
import requests

data_folder = "data"

# Create data folder if it doesn't exist
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# Supported file types and their extensions
supported_file_types = {
    "PDF": ["pdf"],
    "Word": ["docx"],
    "Excel": ["xlsx"],
    "Text": ["txt"],
    "Markdown": ["md"],
    "CSV": ["csv"]
}

# Helper function to save uploaded file
def save_uploaded_file(uploaded_file):
    with open(os.path.join(data_folder, uploaded_file.name), "wb") as f:
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

    # File uploader
    uploaded_files = st.file_uploader("Upload files", type=[ext for exts in supported_file_types.values() for ext in exts], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_uploaded_file(uploaded_file)
        st.success("Files uploaded and saved successfully!")

    # List files in data folder
    existing_files = list(Path(data_folder).glob("*"))
    if existing_files:
        st.write("### Files in Data Folder")
        files_to_delete = []
        for file in existing_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(file.name)
            with col2:
                if st.button(f"Delete {file.name}", key=f"delete_{file.name}"):
                    files_to_delete.append(file)

        # Delete selected files
        for file in files_to_delete:
            os.remove(file)
            st.warning(f"Deleted {file.name}")

    # Process all files in data folder
    if existing_files:
        st.write("### Extracted Content from Files")
        all_text = ""
        for file in existing_files:
            all_text += f"\n---\n**{file.name}:**\n"
            all_text += extract_text_from_file(file)

        st.text_area("Processed Text", all_text, height=300)

    # Chat interface
    user_input = st.text_input("Ask a question about the files")
    if st.button("Send") and user_input:
        # Prepare the prompt
        prompt = f"You are a helpful assistant. The following is the content of the uploaded files:\n\n{all_text}\n\nAnswer the following question:\n{user_input}"

        # Send the prompt to the Ollama server with streaming enabled
        url = "http://localhost:11434/api/generate"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "mistral",  # Replace with your model name
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
                                response_placeholder.markdown(f"**Assistant (streaming):** {full_response}")  # Update with Markdown formatting

                            except json.JSONDecodeError as e:
                                st.error(f"Failed to parse JSON part: {e}")
                                st.write("Problematic JSON part:", line)

                    # Update the chat history with the final response
                    if 'history' not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.append({"role": "assistant", "content": full_response})
                else:
                    st.error(f"Error communicating with Ollama server: {response.status_code}")
                    st.error(response.text)  # Print server error message to debug
        except requests.exceptions.RequestException as e:
            st.error(f"Error communicating with Ollama server: {e}")

    # Display chat history
    if 'history' in st.session_state:
        for chat in st.session_state.history:
            if chat['role'] == 'user':
                st.write(f"**You:** {chat['content']}")
            elif chat['role'] == 'assistant':
                st.write(f"**Assistant:** {chat['content']}")

if __name__ == "__main__":
    main()