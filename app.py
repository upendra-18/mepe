import streamlit as st
import numpy as np
from gradio_client import Client
import tempfile

# -----------------------
# CONFIG
# -----------------------

HF_SPACE_ID = "upendrareddy1/mepe"
HF_TOKEN = st.secrets["HF_TOKEN"]

# Gradio client (HF Space)
hf_client = Client(HF_SPACE_ID)



# -----------------------
# Call HF Space
# -----------------------

from gradio_client import handle_file

def get_persona_embedding(text, image_bytes):

    try:
        # Save image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(image_bytes)
            temp_path = tmp.name

        # Call HF Space API (USE handle_file)
        result = hf_client.predict(
            text=text,
            image=handle_file(temp_path),   # <-- THIS IS THE ONLY CHANGE
            api_name="/mepe_inference"
        )

        # Your space returns embedding
        persona_vector = result["persona_embedding"]

        return persona_vector, None

    except Exception as e:
        return None, str(e)



# -----------------------
# LLM Response Generator
# -----------------------

import requests

HF_LLM_MODEL = "mistralai/Mistral-7B-Instruct"

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
reply = result[0]["generated_text"]

return reply




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
