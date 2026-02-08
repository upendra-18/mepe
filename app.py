# ================================
# MEPE – STREAMLIT (HF GRADIO CALL)
# ================================

import streamlit as st
import requests
import base64
import time
from PIL import Image
import numpy as np

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(
    page_title="MEPE – Multimodal Emotion Persona Engine",
    layout="centered"
)
st.title("🧠 MEPE – Multimodal Emotion Persona Engine")

TEXT_SPACE = "https://upendrareddy1-mepe-text-emotion-api.hf.space"
FACE_SPACE = "https://upendrareddy1-mepe-face-emotion-api.hf.space"

# -------------------------------
# GRADIO SPACE CALL (CORRECT)
# -------------------------------
def gradio_predict(space_url: str, data: list):
    # Step 1: submit job
    r = requests.post(
        f"{space_url}/gradio_api/call/predict",
        json={"data": data},
        timeout=60
    )
    r.raise_for_status()
    event_id = r.json()["event_id"]

    # Step 2: poll result
    while True:
        time.sleep(0.8)
        r = requests.get(
            f"{space_url}/gradio_api/call/predict/{event_id}",
            timeout=60
        )
        if r.status_code != 200:
            continue

        text = r.text
        if "data:" in text:
            result = text.split("data:")[-1].strip()
            return eval(result)  # Gradio returns Python-like list

# -------------------------------
# MODEL CALLS
# -------------------------------
def predict_text_emotion(text: str) -> str:
    result = gradio_predict(TEXT_SPACE, [text])
    return result[0][0]  # label

def predict_face_emotion(img: Image.Image) -> str:
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img)
    buf = base64.b64encode(arr.tobytes()).decode("utf-8")

    result = gradio_predict(FACE_SPACE, [buf])
    return result[0][0]  # label

# -------------------------------
# FUSION LOGIC (NO SHORTCUTS)
# -------------------------------
def resolve_final_emotion(text_e: str, face_e: str) -> str:
    if text_e == face_e:
        return text_e

    priority = ["sad", "angry", "fear", "disgust"]
    for e in priority:
        if e in (text_e, face_e):
            return e

    return text_e

# -------------------------------
# RESPONSE GENERATION (MEPE CORE)
# -------------------------------
def generate_mepe_response(user_text: str, emotion: str) -> str:
    prompt = f"""
You are an emotionally intelligent assistant.

Detected emotional state: {emotion}
Use this only to guide emotional tone.

Respond directly to the user.
Give 1–2 calm, practical suggestions.
Do NOT repeat the user's message.

User message:
{user_text}
""".strip()

    r = requests.post(
        "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
        json={"inputs": prompt},
        timeout=60
    )

    try:
        return r.json()[0]["generated_text"]
    except:
        return "Take a breath. Give yourself a moment to slow down."

# -------------------------------
# UI (UNCHANGED CAMERA INPUT)
# -------------------------------
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
