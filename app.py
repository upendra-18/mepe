# ================================
# MEPE – Multimodal Emotion Persona Engine
# Streamlit + Hugging Face APIs
# ================================

import os
import requests
import streamlit as st
import numpy as np
from PIL import Image
import base64

# -------------------------------
# CONFIG
# -------------------------------
HF_TOKEN = st.secrets.get("HF_TOKEN",None)

TEXT_EMOTION_MODEL = "upendrareddy1/mepe-text-emotion-api"
FACE_EMOTION_MODEL = "upendrareddy1/mepe-face-emotion-api"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(
    page_title="MEPE – Multimodal Emotion Persona Engine",
    layout="centered"
)
st.title("🧠 MEPE – Multimodal Emotion Persona Engine")

# -------------------------------
# API CALLS
# -------------------------------
def predict_text_emotion(text: str) -> str:
    payload = {
        "data": [text]
    }

    r = requests.post(
        "https://upendrareddy1-mepe-text-emotion-api.hf.space/run/predict",
        json=payload,
        timeout=60
    )

    r.raise_for_status()

    # Gradio returns: {"data": [[label, score]]}
    return r.json()["data"][0][0]



def predict_face_emotion(img: Image.Image) -> str:
    img = img.convert("RGB").resize((224, 224))
    buf = base64.b64encode(
        np.array(img).tobytes()
    ).decode("utf-8")

    payload = {"inputs": buf}
    r = requests.post(
        f"https://api-inference.huggingface.co/models/{FACE_EMOTION_MODEL}",
        headers=HEADERS,
        json=payload,
        timeout=60
    )
    r.raise_for_status()
    return r.json()[0]["label"]


def resolve_final_emotion(text_emotion: str, face_emotion: str) -> str:
    if text_emotion == face_emotion:
        return text_emotion

    # Negative emotions override text
    if face_emotion in {"angry", "fear", "sad"}:
        return face_emotion

    return text_emotion


def generate_mepe_response(user_text: str, final_emotion: str) -> str:
    prompt = f"""
You are MEPE — an emotionally intelligent assistant.

The user's inferred emotional state is: {final_emotion}.
Use this only to guide emotional tone.

Rules:
- Respond naturally and directly
- Do NOT repeat the user's message
- Do NOT mention emotion labels unless needed
- Be empathetic and practical
- Give at most 1–2 actionable suggestions

User message:
{user_text}
""".strip()

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 180,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }

    r = requests.post(
        f"https://api-inference.huggingface.co/models/{LLM_MODEL}",
        headers=HEADERS,
        json=payload,
        timeout=120
    )
    r.raise_for_status()

    return r.json()[0]["generated_text"].strip()

# -------------------------------
# UI FLOW
# -------------------------------
user_text = st.text_area("How are you feeling?")
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
            response = generate_mepe_response(user_text, final_emotion)

        st.subheader("🧭 Inferred Emotional State")
        st.write(final_emotion)

        st.subheader("💬 MEPE Response")
        st.write(response)
