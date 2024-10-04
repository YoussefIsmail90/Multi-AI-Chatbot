import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import whisper
import tempfile

# Function to describe the uploaded image
def describe_image(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    image = Image.open(image_path)
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    return description

# Function to generate a story from the image description
def generate_story(description):
    try:
        generator = pipeline('text-generation', model='gpt2')
        story = generator(description, max_length=300, num_return_sequences=1)
        return story[0]['generated_text']
    except EnvironmentError as e:
        return f"Environment error: {str(e)}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Function to transcribe audio using Whisper (using the tiny model)
def transcribe_audio(audio_file_path):
    model = whisper.load_model("tiny")  # Load the tiny Whisper model
    result = model.transcribe(audio_file_path)
    return result['text']

# Streamlit app
st.title("Image to Story Converter with Audio Transcription")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button("Generate Story"):
        with st.spinner("Generating story..."):
            description = describe_image(uploaded_file)
            st.write(f"Image Description: {description}")

            story = generate_story(description)
            st.write(f"Generated Story: {story}")

            # Provide an option to upload an audio file for transcription
            audio_file = st.file_uploader("Upload an audio file for transcription", type=["mp3", "wav"])
            if audio_file is not None:
                with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
                    temp_audio_file.write(audio_file.read())
                    temp_audio_file_path = temp_audio_file.name
                transcription = transcribe_audio(temp_audio_file_path)
                st.write(f"Transcription: {transcription}")
