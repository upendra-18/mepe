# ================================
# MEPE – STABLE INFERENCE VERSION
# ================================

import streamlit as st
import requests
import base64
import io
import numpy as np
from PIL import Image

# -------------------------------
# CONFIG
# -------------------------------
TEXT_MODEL = "upendrareddy1/mepe-text-emotion"
FACE_MODEL = "upendrareddy1/mepe-face-emotion"

TEXT_API = f"https://api-inference.huggingface.co/models/{TEXT_MODEL}"
FACE_API = f"https://api-inference.huggingface.co/models/{FACE_MODEL}"

HEADERS = {"Content-Type": "application/json"}

# -------------------------------
# HELPERS
# -------------------------------
def predict_text_emotion(text: str) -> str:
    r = requests.post(
        TEXT_API,
        headers=HEADERS,
        json={"inputs": text},
        timeout=30
    )
    r.raise_for_status()
    return r.json()[0]["label"]


def predict_face_emotion(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    r = requests.post(
        FACE_API,
        headers=HEADERS,
        json={"inputs": f"data:image/jpeg;base64,{b64}"},
        timeout=30
    )
    r.raise_for_status()
    return r.json()[0]["label"]


def resolve_final_emotion(text_e: str, face_e: str) -> str:
    # simple deterministic rule (NO bullshit fusion)
    if text_e == face_e:
        return text_e
    return text_e  # text has priority (as you designed earlier)


def generate_response(user_text: str, emotion: str) -> str:
    return (
        f"I sense **{emotion}** in how you’re feeling.\n\n"
        "Take a breath. Try one small action today that gives you control—"
        "a short walk, writing your thoughts, or stepping away from the screen."
    )

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(
    page_title="MEPE – Emotion Aware AI",
    layout="centered"
)

st.title("🧠 MEPE – Multimodal Emotion Persona Engine")

user_text = st.text_area("How are you feeling right now?")
image = st.camera_input("Capture your facial expression")

if st.button("Analyze & Respond"):
    if not user_text or image is None:
        st.warning("Both text and face input are required.")
    else:
        img = Image.open(image)

        with st.spinner("Understanding you..."):
            text_emotion = predict_text_emotion(user_text)
            face_emotion = predict_face_emotion(img)
            final_emotion = resolve_final_emotion(text_emotion, face_emotion)
            response = generate_response(user_text, final_emotion)

        st.subheader("🧭 Inferred Emotional State")
        st.write(final_emotion)

        st.subheader("💬 MEPE Response")
        st.write(response)
