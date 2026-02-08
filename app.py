# ================================
# MEPE – STREAMLIT (STABLE)
# Inference API (NO SPACES)
# ================================

import streamlit as st
import requests
import base64
import io
from PIL import Image

# -------------------------------
# CONFIG
# -------------------------------
HF_TOKEN = st.secrets["HF_TOKEN"]

TEXT_MODEL = "upendrareddy1/mepe-text-emotion"
FACE_MODEL = "upendrareddy1/mepe-face-emotion"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

TEXT_API = f"https://api-inference.huggingface.co/models/{TEXT_MODEL}"
FACE_API = f"https://api-inference.huggingface.co/models/{FACE_MODEL}"

# -------------------------------
# API CALLS (SYNC, SAFE)
# -------------------------------
def predict_text_emotion(text: str) -> str:
    r = requests.post(
        TEXT_API,
        headers=HEADERS,
        json={"inputs": text},
        timeout=20
    )
    r.raise_for_status()
    return r.json()[0]["label"]

def predict_face_emotion(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    r = requests.post(
        FACE_API,
        headers=HEADERS,
        json={"inputs": f"data:image/jpeg;base64,{b64}"},
        timeout=20
    )
    r.raise_for_status()
    return r.json()[0]["label"]

# -------------------------------
# MEPE LOGIC (UNCHANGED)
# -------------------------------
def resolve_final_emotion(text_e, face_e):
    if text_e == face_e:
        return text_e
    if face_e in ["sad", "angry", "fear"]:
        return face_e
    return text_e

def generate_mepe_response(user_text, emotion):
    return f"""
I sense **{emotion}** from your expression and words.

Let’s slow down for a moment.
Try taking one deep breath and focus on one small thing you can control right now.

I’m here with you.
""".strip()

# -------------------------------
# STREAMLIT UI (CAMERA)
# -------------------------------
st.set_page_config(
    page_title="MEPE – Multimodal Emotion Persona Engine",
    layout="centered"
)

st.title("🧠 MEPE – Multimodal Emotion Persona Engine")

user_text = st.text_area("How are you feeling?")
image = st.camera_input("Capture your facial expression")

if st.button("Analyze & Respond"):
    if not user_text or image is None:
        st.warning("Both text and face input are required.")
    else:
        img = Image.open(image)

        with st.spinner("Understanding you..."):
            text_e = predict_text_emotion(user_text)
            face_e = predict_face_emotion(img)
            final_e = resolve_final_emotion(text_e, face_e)
            response = generate_mepe_response(user_text, final_e)

        st.subheader("🧭 Inferred Emotional State")
        st.write(final_e)

        st.subheader("💬 MEPE Response")
        st.write(response)
