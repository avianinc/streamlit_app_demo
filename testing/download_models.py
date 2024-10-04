import streamlit as st
import requests
import logging
import json

# Initialize Streamlit
st.set_page_config(
    page_title="Ollama Model Downloader",
    page_icon="⬇️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Logging Configuration
logging.basicConfig(
    filename='download_app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Function to stream model download progress from Ollama server
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
    st.title("Ollama Model Downloader")

    # Input for model name
    model_name = st.text_input("Enter the model name to download", placeholder="e.g., llama3.2")

    # Download button
    if st.button("Download Model"):
        if model_name.strip():
            download_model(model_name.strip())
        else:
            st.warning("Please enter a model name.")

if __name__ == "__main__":
    main()
