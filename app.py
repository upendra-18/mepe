import streamlit as st
import requests
import numpy as np
import openai

# -----------------------
# CONFIG
# -----------------------

HF_SPACE_URL = "https://huggingface.co/spaces/upendrareddy1/mepe"
OPENAI_API_KEY = "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"

openai.api_key = OPENAI_API_KEY

# -----------------------
# Call HF Space
# -----------------------

def get_persona_embedding(text, image_bytes):

    files = {
        "data": (
            None,
            str([text]),  # text input
        ),
        "files": (
            "image.png",
            image_bytes,
            "image/png"
        )
    }

    response = requests.post(HF_SPACE_URL, files=files)

    if response.status_code != 200:
        return None, response.text

    result = response.json()

    # Adjust based on your space output structure
    persona_vector = result["data"][0]["persona_embedding"]

    return persona_vector, None


# -----------------------
# LLM Response Generator
# -----------------------

def generate_response(persona_vector, user_text):

    prompt = f"""
You are an emotionally intelligent assistant.

Persona vector (512-dim):
{persona_vector}

User message:
{user_text}

Generate a supportive, emotionally aligned response.
Keep it natural and human.
"""

    completion = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are emotionally intelligent."},
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content


# -----------------------
# Streamlit UI
# -----------------------

st.set_page_config(page_title="MEPE", layout="centered")

st.title("🧠 MEPE – Multimodal Emotion Persona Engine")

text_input = st.text_area("Enter your message")
image_input = st.file_uploader("Upload face image", type=["png", "jpg", "jpeg"])

if st.button("Generate Response"):

    if not text_input or not image_input:
        st.error("Both text and image required.")
    else:

        with st.spinner("Getting persona embedding..."):

            image_bytes = image_input.read()
            persona_vector, error = get_persona_embedding(text_input, image_bytes)

            if error:
                st.error(error)
            else:

                with st.spinner("Generating response..."):
                    reply = generate_response(persona_vector, text_input)

                st.success("Response generated.")
                st.write(reply)
