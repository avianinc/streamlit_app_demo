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

            # Send the prompt to the Ollama server with streaming enabled
            url = "http://localhost:11434/api/generate"  # Using the endpoint that worked with curl
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
                                    response_placeholder.markdown(f"**Assistant:** {full_response}")  # Update with Markdown formatting

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

    # Display chat history
    for chat in st.session_state.history:
        if chat['role'] == 'user':
            st.write(f"**You:** {chat['content']}")
        elif chat['role'] == 'assistant':
            # Skip printing the final response again, since it was already displayed during streaming
            if chat['content'] != full_response:
                st.write(f"**Assistant:** {chat['content']}")
