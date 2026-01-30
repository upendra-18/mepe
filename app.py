# ================================
# MEPE – FINAL CLEAN STABLE APP
# ================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import subprocess
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from transformers import AutoTokenizer, TFDistilBertModel

# -------------------------------
# Streamlit config
# -------------------------------
st.set_page_config(
    page_title="MEPE – Multimodal Emotion Persona Engine",
    layout="centered"
)
st.title("🧠 MEPE – Multimodal Emotion Persona Engine")

# -------------------------------
# Load models (CACHED)
# -------------------------------
@st.cache_resource
def load_models():
    # Text encoder
    tokenizer = AutoTokenizer.from_pretrained(
        "upendrareddy1/mepe-text-emotion"
    )
    text_encoder = TFDistilBertModel.from_pretrained(
        "upendrareddy1/mepe-text-emotion"
    )
    text_encoder.trainable = False

    # Face emotion classifier (7 classes)
    face_model = tf.keras.models.load_model(
        "models/face_emotion/model.keras",
        compile=False,
        safe_mode=False
    )

    return tokenizer, text_encoder, face_model


tokenizer, text_encoder, face_model = load_models()

# -------------------------------
# Helpers
# -------------------------------
def text_embedding(text: str) -> np.ndarray:
    tokens = tokenizer(
        text,
        return_tensors="tf",
        truncation=True,
        padding=True,
        max_length=128
    )
    outputs = text_encoder(**tokens)
    emb = tf.reduce_mean(outputs.last_hidden_state, axis=1)
    return emb.numpy().squeeze()


def face_emotion(img: Image.Image) -> str:
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype="float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    probs = face_model.predict(arr, verbose=0)[0]

    emotions = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "sad",
        "surprise",
        "neutral"
    ]
    return emotions[int(np.argmax(probs))]


import subprocess

OLLAMA_EXE = r"C:\Users\upend\AppData\Local\Programs\Ollama\ollama.exe"

def generate_response(user_text: str, face_label: str) -> str:
    prompt = f"""
You are an emotionally intelligent assistant.

The user's facial expression suggests: {face_label}.
Use this only to guide emotional tone.

Respond directly to the user.
Offer calm, practical emotional support.
Give one or two actionable suggestions.

User message:
{user_text}
""".strip()

    result = subprocess.run(
        [OLLAMA_EXE, "run", "mistral"],
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8"
    )

    output = result.stdout.strip()

    if not output:
        output = "Take a moment to rest and allow yourself to slow down today."

    return output



# -------------------------------
# UI
# -------------------------------
user_text = st.text_area("How are you feeling?")
image = st.camera_input("Capture your facial expression")

if st.button("Analyze & Respond"):
    if not user_text or image is None:
        st.warning("Both inputs are required.")
    else:
        img = Image.open(image)

        # perception
        _ = text_embedding(user_text)   # semantic understanding (internal use)
        face_label = face_emotion(img)

        # response generation
        response = generate_response(user_text, face_label)

        st.subheader("Detected Facial Emotion")
        st.write(face_label)

        st.subheader("Response")
        st.write(response)
