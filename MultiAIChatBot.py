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
        generator = pipeline('text-generation', model='gpt-2')
        story = generator(description, max_length=150, num_return_sequences=1)
        return story[0]['generated_text']
    except EnvironmentError as e:
        return f"Environment error: {str(e)}"
    except PipelineException as e:
        return f"Pipeline error: {str(e)}"
    except Exception as e:
        return f"An error occurred: {str(e)}"


# Function to convert the story to audio
def convert_text_to_audio(text):
    tts = gTTS(text)
    audio_file = "story.mp3"
    tts.save(audio_file)
    return audio_file

# Streamlit app
st.title("Image to Story Converter")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button("Generate Story"):
        description = describe_image(uploaded_file)
        st.write(f"Image Description: {description}")

        story = generate_story(description)
        st.write(f"Generated Story: {story}")
        
        audio_file = convert_text_to_audio(story)
        st.audio(audio_file)
