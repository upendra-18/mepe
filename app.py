# ==========================================
# MEPE – Emotion Aware AI (FINAL)
# Streamlit + Hugging Face Gradio Spaces
# ==========================================

import time
import base64
import requests
import numpy as np
import streamlit as st
from PIL import Image

# ==========================================
# CONFIG
# ==========================================

st.set_page_config(
    page_title="MEPE – Emotion Aware AI",
    page_icon="🧠",
    layout="centered"
)

TEXT_API = "https://upendrareddy1-mepe-text-emotion-api.hf.space"
FACE_API = "https://upendrareddy1-mepe-face-emotion-api.hf.space"

# ==========================================
# GRADIO CLIENT (CORRECT IMPLEMENTATION)
# ==========================================

def gradio_predict(base_url: str, data: list):
    """
    Correct way to call Hugging Face Gradio Spaces.
    1. POST -> get event_id
    2. Poll -> get streamed result
    """

    # Step 1: Start prediction
    r = requests.post(
        f"{base_url}/gradio_api/call/predict",
        json={"data": data},
        timeout=60
    )
    r.raise_for_status()

    event_id = r.json()["event_id"]

    # Step 2: Poll result
    result_url = f"{base_url}/gradio_api/call/predict/{event_id}"

    while True:
        res = requests.get(result_url, timeout=60)
        res.raise_for_status()

        for line in res.text.splitlines():
            if line.startswith("data:"):
                payload = line.replace("data:", "").strip()

                if payload == "[DONE]":
                    break

                result = eval(payload)
                return result["data"]

        time.sleep(0.4)

# ==========================================
# MODEL CALLS
# ==========================================

def predict_text_emotion(text: str) -> str:
    result = gradio_predict(TEXT_API, [text])
    return result[0][0]   # label

def predict_face_emotion(img: Image.Image) -> str:
    img = img.convert("RGB").resize((224, 224))
    buf = base64.b64encode(np.array(img).tobytes()).decode("utf-8")
    result = gradio_predict(FACE_API, [buf])
    return result[0][0]

# ==========================================
# EMOTION FUSION LOGIC (CORE MEPE)
# ==========================================

def resolve_final_emotion(text_e: str, face_e: str) -> str:
    """
    Conflict-aware fusion logic
    """
    if text_e == face_e:
        return text_e

    dominance = {
        "angry": 3,
        "fear": 3,
        "sad": 2,
        "disgust": 2,
        "happy": 1,
        "surprise": 1,
        "neutral": 0
    }

    return text_e if dominance.get(text_e, 0) >= dominance.get(face_e, 0) else face_e

# ==========================================
# PERSONA-AWARE RESPONSE GENERATION
# ==========================================

def generate_mepe_response(user_text: str, emotion: str) -> str:
    responses = {
        "happy": "That’s great to hear 😊 Keep that positive momentum going.",
        "sad": "I sense you’re feeling low. Want to talk about what’s weighing on you?",
        "angry": "I can feel the frustration. Let’s slow this down and think clearly.",
        "fear": "It sounds like something is worrying you. You’re not alone here.",
        "disgust": "That reaction makes sense. Want to unpack what triggered it?",
        "surprise": "That caught you off guard, didn’t it? Tell me more.",
        "neutral": "I’m here. What would you like to explore next?"
    }

    return responses.get(emotion, "I’m here with you. Tell me more.")

# ==========================================
# STREAMLIT UI
# ==========================================

st.title("🧠 MEPE – Emotion Aware AI")

st.markdown(
    """
MEPE understands **how you feel**, not just **what you say**.
It combines **text emotion**, **facial emotion**, and **fusion logic**
to respond in a human-aware way.
"""
)

user_text = st.text_area("💬 Type your message", height=120)
image = st.file_uploader("📷 Upload a face image", type=["jpg", "png", "jpeg"])

if st.button("Analyze & Answer"):
    if not user_text or not image:
        st.warning("Please provide both text and an image.")
    else:
        img = Image.open(image)

        with st.spinner("Understanding you..."):
            text_e = predict_text_emotion(user_text)
            face_e = predict_face_emotion(img)
            final_e = resolve_final_emotion(text_e, face_e)
            response = generate_mepe_response(user_text, final_e)

        st.subheader("🧭 Emotion Analysis")
        st.write(f"**Text Emotion:** {text_e}")
        st.write(f"**Face Emotion:** {face_e}")
        st.write(f"**Final Emotion:** {final_e}")

        st.subheader("💬 MEPE Response")
        st.write(response)
