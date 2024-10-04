import streamlit as st
import logging
from openai import OpenAI
from pathlib import Path
import ollama

# Initialize the OpenAI client
client = OpenAI(base_url="http://ollama:11434/v1", api_key="ollama")

# Initialize session history if it doesn't exist
if 'history' not in st.session_state:
    st.session_state.history = []

# Chat history UI
for chat in st.session_state.history:
    role = "assistant" if chat["role"] == "assistant" else "user"
    avatar = "🤖" if role == "assistant" else "😎"
    with st.chat_message(role, avatar=avatar):
        st.markdown(chat["content"])

# User input for chat
user_input = st.chat_input("Ask a question about the files")

# If the user submits a question and there are models available
if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    logging.info(f"User input: {user_input}")

    # Example of processing data (you can replace with your data handling logic)
    context = "This is your retrieved context based on uploaded data."  # Replace with actual context fetching

    # Prepare the prompt
    prompt = (
        f"You are a helpful assistant. Answer the following question using the provided context. "
        f"Do not mention the context or search results in your answer. "
        f"Context:\n{context}\n\n"
        f"Question:\n{user_input}"
    )

    # Send the request to the model (streaming enabled)
    try:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response_text = ""

            # Stream response from Ollama
            response = client.completions.create(
                model="mistral:latest",  # Assuming a default model
                prompt=prompt,
                stream=True
            )

            for chunk in response:
                response_chunk = chunk.choices[0].text
                response_text += response_chunk
                message_placeholder.markdown(response_text)

        # Append the final assistant response to the chat history
        st.session_state.history.append({"role": "assistant", "content": response_text})
        logging.info(f"Assistant response: {response_text}")

    except Exception as e:
        st.error(f"Error occurred while generating response: {e}")
        logging.error(f"Error occurred while generating response: {e}")
