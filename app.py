import streamlit as st
import numpy as np
from gradio_client import Client, handle_file
import tempfile
import requests

# -----------------------
# CONFIG
# -----------------------

HF_SPACE_ID = "upendrareddy1/mepe"
HF_TOKEN = st.secrets["HF_TOKEN"]

HF_LLM_MODEL = "HuggingFaceH4/zephyr-7b-beta"


# Initialize HF Space client
hf_client = Client(HF_SPACE_ID)


# -----------------------
# Call HF Space
# -----------------------

def get_persona_embedding(text, image_bytes):

    try:
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(image_bytes)
            temp_path = tmp.name

        # Call HF Space API
        result = hf_client.predict(
            text=text,
            image=handle_file(temp_path),
            api_name="/mepe_inference"
        )

        # Extract embedding
        persona_vector = result["persona_embedding"]

        return persona_vector, None

    except Exception as e:
        return None, str(e)


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

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.7
        }
    }

    response = requests.post(
        f"https://router.huggingface.co/hf-inference/models/{HF_LLM_MODEL}",
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        return f"LLM Error: {response.text}"

    result = response.json()

    # Mistral returns list format
    return result[0]["generated_text"]


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
