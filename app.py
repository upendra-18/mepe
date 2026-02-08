import streamlit as st
import requests
import numpy as np
from PIL import Image
import io

# ==============================
# CONFIG — UPDATE ONLY URLs
# ==============================

TEXT_EMOTION_API = (
    "https://api-inference.huggingface.co/models/"
    "upendrareddy1/mepe-text-emotion-api"
)

FACE_EMOTION_API = (
    "https://api-inference.huggingface.co/models/"
    "upendrareddy1/mepe-face-emotion-api"
)

CRITICAL_FACE_EMOTIONS = {"fear", "anger", "sad"}

# ==============================
# HELPERS — API CALLS
# ==============================

def call_text_emotion(input_ids, attention_mask):
    payload = {
        "inputs": {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    }
    r = requests.post(TEXT_EMOTION_API, json=payload, timeout=120)
    r.raise_for_status()

    # Expected: embedding or logits
    return r.json()


def call_face_emotion(image: Image.Image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    r = requests.post(
        FACE_EMOTION_API,
        files={"file": buf},
        timeout=120
    )
    r.raise_for_status()

    # Expected: {"label": "..."} or logits
    return r.json()


def resolve_final_emotion(text_emotion: str, face_emotion: str) -> str:
    # Rule 1: agreement
    if text_emotion == face_emotion:
        return text_emotion

    # Rule 2: critical face override
    if face_emotion in CRITICAL_FACE_EMOTIONS:
        return face_emotion

    # Rule 3: default to text
    return text_emotion


# ==============================
# STREAMLIT UI
# ==============================

st.set_page_config(
    page_title="MEPE – Multimodal Emotion Engine",
    layout="centered"
)

st.title("🧠 MEPE – Multimodal Emotion Engine")

st.markdown("### Inputs")

user_text = st.text_input("User text")
image = st.camera_input("Capture facial expression")

if st.button("Analyze"):
    if not user_text or image is None:
        st.warning("Both text and image are required.")
        st.stop()

    # ⚠️ TEMP tokens (replace with tokenizer later)
    input_ids = [101, 1045, 2572, 1037, 2204, 2154, 102]
    attention_mask = [1, 1, 1, 1, 1, 1, 1]

    with st.spinner("Running emotion analysis..."):
        # ---- Text ----
        text_result = call_text_emotion(input_ids, attention_mask)

        # TODO: map output → emotion label
        # Placeholder until classifier is wired
        text_emotion = "neutral"

        # ---- Face ----
        face_result = call_face_emotion(Image.open(image))

        # Expecting {"label": "..."}
        face_emotion = face_result.get("label", "neutral")

        # ---- Final decision ----
        final_emotion = resolve_final_emotion(
            text_emotion,
            face_emotion
        )

    st.subheader("Results")
    st.write("**Text Emotion:**", text_emotion)
    st.write("**Face Emotion:**", face_emotion)
    st.success(f"🎯 Final Emotion: {final_emotion}")
