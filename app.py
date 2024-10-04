import streamlit as st
import requests
import pdfplumber
import json

st.title("PDF Chat with Ollama")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from PDF using pdfplumber
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        max_pages = 5  # Limit to first 5 pages
        for page in pdf.pages[:max_pages]:
            text += page.extract_text() or ""  # Handle cases where text extraction fails

    st.write("PDF content extracted.")

    # Initialize session state for chat history
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Chat interface
    user_input = st.text_input("Ask a question about the PDF")
    if st.button("Send"):
        if user_input:
            # Add user's question to history
            st.session_state.history.append({"role": "user", "content": user_input})

            # Prepare the prompt
            prompt = (
                "You are a helpful assistant. The following is the content of a PDF document:\n\n"
                + text
                + "\n\nAnswer the following question:\n"
                + user_input
            )

            # Send the prompt to the Ollama server
            url = "http://localhost:11434/api/generate"  # Using the endpoint that worked with curl
            headers = {"Content-Type": "application/json"}
            data = {
                "model": "mistral",  # Replace with your model name
                "prompt": prompt
            }
            try:
                response = requests.post(url, json=data, headers=headers)

                # Read response content as text first
                response_text = response.text

                if response.status_code == 200:
                    try:
                        # Split the response text into separate JSON objects
                        json_objects = response_text.splitlines()
                        answer = ""

                        for json_obj in json_objects:
                            try:
                                # Load each JSON object and accumulate the response
                                parsed_obj = json.loads(json_obj)
                                answer += parsed_obj.get("response", "")
                            except json.JSONDecodeError as e:
                                st.error(f"Failed to parse JSON part: {e}")
                                st.write("Problematic JSON part:", json_obj)

                        # Add the accumulated response to history
                        st.session_state.history.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Failed to process response: {e}")
                        st.write("Response text:", response_text)  # Display full response to debug
                else:
                    st.error(f"Error communicating with Ollama server: {response.status_code}")
                    st.error(response_text)  # Print server error message to debug
            except requests.exceptions.RequestException as e:
                st.error(f"Error communicating with Ollama server: {e}")

    # Display chat history
    for chat in st.session_state.history:
        if chat['role'] == 'user':
            st.write("**You:** " + chat['content'])
        else:
            st.write("**Assistant:** " + chat['content'])
