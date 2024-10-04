import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from gtts import gTTS
import os

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
        generator = pipeline('text-generation', model='gpt2')  # GPT-2 is primarily English, but we can use a prompt in Arabic.
        arabic_description = "أخبرني قصة عن: " + description  # Create a prompt in Arabic
        story = generator(arabic_description, max_length=300, num_return_sequences=1)
        return story[0]['generated_text']
    except EnvironmentError as e:
        return f"Environment error: {str(e)}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Function to convert text to audio in Arabic
def text_to_audio(text):
    tts = gTTS(text=text, lang='ar')  # Specify Arabic language
    audio_file_path = "generated_story.mp3"  # Output audio file path
    tts.save(audio_file_path)
    return audio_file_path

# Streamlit app
st.title("Image to Story Converter with Text-to-Audio")  # Title in English

# File uploader prompt in English
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Button to generate the story
    if st.button("Generate Story"):
        with st.spinner("Generating story..."):
            description = describe_image(uploaded_file)
            st.write(f"Image Description: {description}")

            story = generate_story(description)
            st.write(f"Generated Story: {story}")

            # Convert the story to audio
            audio_file_path = text_to_audio(story)
            st.audio(audio_file_path)  # Play the audio

            # Clean up the generated audio file if needed
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)
