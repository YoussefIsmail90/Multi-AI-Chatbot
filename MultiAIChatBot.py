import streamlit as st
import whisper
from transformers import pipeline

# Load models
whisper_model = whisper.load_model("large")
llama_model = pipeline('text-generation', model="meta-llama/Llama-3.2-1B")

# Title of the app
st.title("Multimodal Chatbot")

# Sidebar for choosing the task
option = st.sidebar.selectbox(
    "Choose an option:",
    ("Text Chat", "Voice Chat")
)

# Text Chat
if option == "Text Chat":
    st.subheader("Text Chat")
    user_input = st.text_input("Type your question:")
    if user_input:
        st.write("Generating response...")
        response = llama_model(user_input)
        st.write(f"Chatbot: {response[0]['generated_text']}")

# Voice Chat
elif option == "Voice Chat":
    st.subheader("Voice Chat")
    st.write("Upload an audio file, and the chatbot will respond.")
    audio_input = st.file_uploader("Upload your audio file", type=["wav", "mp3"])
    if audio_input:
        st.write("Transcribing audio...")
        transcription = whisper_model.transcribe(audio_input)
        st.write(f"Transcription: {transcription['text']}")
        user_input = transcription["text"]
        if user_input:
            st.write("Generating response...")
            response = llama_model(user_input)
            st.write(f"Chatbot: {response[0]['generated_text']}")
