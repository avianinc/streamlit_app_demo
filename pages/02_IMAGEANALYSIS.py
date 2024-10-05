import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO
import json
import logging

# Initialize Streamlit
st.set_page_config(
    page_title="Sandbox",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levellevelname)s - %(message)s',
    level=logging.INFO
)

def img_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def get_allowed_model_names(models_info):
    allowed_models = ["bakllava:latest", "llava:latest"]
    return tuple(
        model
        for model in allowed_models
        if model in [m["id"] for m in models_info["data"]]
    )

def download_model_with_progress(model_name):
    API_URL = "http://ollama:11434/api/pull"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    data = {"name": model_name}
    
    try:
        response = requests.post(API_URL, json=data, headers=headers, stream=True)
        progress_bar = st.progress(0)
        progress_text = st.empty()
        total_size = 0
        downloaded_size = 0

        if response.status_code == 200:
            for line in response.iter_lines(decode_unicode=True):
                if line.strip():
                    progress_data = json.loads(line)
                    if "total" in progress_data and "completed" in progress_data:
                        total_size = progress_data["total"]
                        downloaded_size = progress_data["completed"]
                        percent_complete = int((downloaded_size / total_size) * 100)
                        progress_bar.progress(percent_complete)
                        progress_text.text(f"Downloading: {downloaded_size}/{total_size} bytes ({percent_complete}%)")
                        
                    elif progress_data.get("status") == "success":
                        progress_bar.progress(100)
                        progress_text.text(f"Download complete: {model_name}")
                        st.success(f"Model '{model_name}' downloaded successfully.")
                        break
        else:
            st.error(f"Failed to pull model '{model_name}': {response.status_code}")
    except Exception as e:
        st.error(f"Error during model download: {str(e)}")

def main():
    st.subheader("Image Analysis Sandbox", anchor=False)

    # Get the available models
    try:
        models_info = requests.get("http://ollama:11434/v1/models").json()
        available_models = get_allowed_model_names(models_info)
        missing_models = set(["bakllava:latest", "llava:latest"]) - set(available_models)
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
        available_models = []
        missing_models = set(["bakllava:latest", "llava:latest"])

    col_1, col_2 = st.columns(2)
    with col_1:
        if not available_models:
            st.error("No allowed models are available.")
            model_to_download = st.selectbox(
                "Select a model to download", ["bakllava:latest", "llava:latest"]
            )
            if st.button(f"Download {model_to_download}"):
                download_model_with_progress(model_to_download)
                st.rerun()  # Only rerun here when model download is done
        else:
            if missing_models:
                model_to_download = st.selectbox("Download missing model", list(missing_models))
                if st.button(f"Download {model_to_download}"):
                    download_model_with_progress(model_to_download)
                    st.rerun()  # Rerun when model download is complete

            selected_model = st.selectbox("Delete model", available_models)
            if st.button(f"Delete {selected_model}"):
                try:
                    response = requests.delete(f"http://ollama:11434/v1/models/{selected_model}")
                    st.success(f"Deleted model: {selected_model}")
                    st.rerun()  # Trigger a rerun after model deletion
                except Exception as e:
                    st.error(f"Failed to delete model: {selected_model}. Error: {str(e)}")

    if not available_models:
        return

    selected_model = col_2.selectbox("Pick a model available locally on your system", available_models, key="model_picker")

    if "chats" not in st.session_state:
        st.session_state.chats = []
    
    # To avoid the repeated responses issue, add a flag to track if the request is processing
    if "response_processed" not in st.session_state:
        st.session_state.response_processed = False

    uploaded_file = st.file_uploader("Upload an image for analysis", type=["png", "jpg", "jpeg"])

    col1, col2 = st.columns(2)
    with col2:
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded image", use_column_width=True)

    with col1:
        if uploaded_file:
            # Display chat history
            for message in st.session_state.chats:
                avatar = "🌋" if message["role"] == "assistant" else "🫠"
                st.markdown(f"**{message['role'].capitalize()}**: {message['content']}")

            # Handle user input
            user_input = st.text_input("Question about the image...", key="user_input")
            submit = st.button("Submit")

            if submit and user_input and not st.session_state.response_processed:
                st.session_state.chats.append({"role": "user", "content": user_input})
                
                image_base64 = img_to_base64(image)
                API_URL = "http://ollama:11434/api/generate"
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                }
                data = {
                    "model": selected_model,
                    "prompt": user_input,
                    "images": [image_base64],
                }

                st.session_state.chats.append({"role": "assistant", "content": "Processing..."})
                response_text = ""

                with st.spinner("Processing..."):
                    response = requests.post(API_URL, json=data, headers=headers)
                    if response.status_code == 200:
                        response_lines = response.text.split("\n")
                        for line in response_lines:
                            if line.strip():
                                try:
                                    response_data = json.loads(line)
                                    if "response" in response_data:
                                        response_text += response_data["response"]
                                except json.JSONDecodeError:
                                    continue
                    else:
                        response_text = f"Failed to get a response from {selected_model}."

                # Update chat with the assistant's response
                st.session_state.chats[-1]["content"] = response_text
                st.session_state.response_processed = True  # Mark response as processed
                st.rerun()  # Force rerun to display the response

    if st.button("Clear Chat"):
        st.session_state.chats = []
        st.session_state.response_processed = False  # Reset processing flag
        st.rerun()  # Clear everything and rerun the app

if __name__ == "__main__":
    main()
