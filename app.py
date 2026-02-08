import streamlit as st
import requests
import time
import base64
import io
import numpy as np
from PIL import Image

# ================================
# CONFIG
# ================================
TEXT_SPACE = "https://upendrareddy1-mepe-text-emotion-api.hf.space"
FACE_SPACE = "https://upendrareddy1-mepe-face-emotion-api.hf.space"

# ================================
# GRADIO SPACE CALL
# ================================
def gradio_predict(space_url, data):
    submit = requests.post(
        f"{space_url}/gradio_api/call/predict",
        json={"data": data},
        timeout=30
    )
    submit.raise_for_status()
    event_id = submit.json()["event_id"]

    while True:
        poll = requests.get(
            f"{space_url}/gradio_api/call/predict/{event_id}",
            timeout=30
        )
        if poll.status_code != 200:
            time.sleep(0.5)
            continue

        text = poll.text
        if text.startswith("data:"):
            return eval(text.replace("data:", "").strip())

        time.sleep(0.5)

# ================================
# MODEL CALLS
# ================================
def predict_text_emotion(text):
    result = gradio_predict(TEXT_SPACE, [text])
    return result[0][0]

def predict_face_emotion(img: Image.Image):
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    result = gradio_predict(FACE_SPACE, [f"data:image/jpeg;base64,{b64}"])
    return result[0][0]

# ================================
# MEPE LOGIC
# ================================
def resolve_final_emotion(text_e, face_e):
    if text_e == face_e:
        return text_e
    return text_e  # text has priority (intent > expression)

def generate_mepe_response(text, emotion):
    return f"I sense **{emotion}**. Tell me more about what you're feeling."

# ================================
# STREAMLIT UI
# ================================
st.set_page_config(page_title="MEPE", layout="centered")
st.title("🧠 MEPE — Emotion-Aware AI")

user_text = st.text_area("Your message")
image = st.camera_input("Capture your face")

if st.button("Analyze") and user_text and image:
    img = Image.open(image)

    with st.spinner("Understanding you..."):
        text_e = predict_text_emotion(user_text)
        face_e = predict_face_emotion(img)
        final_e = resolve_final_emotion(text_e, face_e)
        response = generate_mepe_response(user_text, final_e)

    st.subheader("🧭 Emotion")
    st.write(final_e)

    st.subheader("💬 MEPE Response")
    st.write(response)
