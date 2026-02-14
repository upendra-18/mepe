import streamlit as st
import numpy as np
from gradio_client import Client
import tempfile
from openai import OpenAI

# -----------------------
# CONFIG
# -----------------------

HF_SPACE_ID = "upendrareddy1/mepe"

# Gradio client (HF Space)
hf_client = Client(HF_SPACE_ID)

# OpenAI client
llm_client = OpenAI(api_key="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")


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

    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are emotionally intelligent."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


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
