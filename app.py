# ================================
# MEPE – LOCAL .KERAS VERSION (FINAL)
# ================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

from transformers import AutoTokenizer, TFDistilBertModel, pipeline

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
    # TEXT
    tokenizer = AutoTokenizer.from_pretrained(
        "upendrareddy1/mepe-text-emotion"
    )
    text_encoder = TFDistilBertModel.from_pretrained(
        "upendrareddy1/mepe-text-emotion"
    )
    text_encoder.trainable = False

    # FACE (.keras – legacy compatible)
    face_model = tf.keras.models.load_model(
        "models/face_emotion/model.keras",
        compile=False,
        safe_mode=False   # <-- THIS IS CRITICAL
    )

    # LLM
    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=-1
    )

    return tokenizer, text_encoder, face_model, llm


tokenizer, text_encoder, face_model, llm = load_models()

# -------------------------------
# Helpers
# -------------------------------
def text_embedding(text):
    tokens = tokenizer(
        text,
        return_tensors="tf",
        truncation=True,
        padding=True,
        max_length=128
    )
    out = text_encoder(**tokens)
    return tf.reduce_mean(out.last_hidden_state, axis=1).numpy()[0]


def face_embedding(img):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype="float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return face_model.predict(arr, verbose=0)[0]


def gated_fusion(t, f):
    a = np.mean(t) / (np.mean(t) + np.mean(f) + 1e-6)
    return a * t + (1 - a) * f


def build_prompt(text):
    return f"""
You are an emotionally intelligent assistant.
Respond with empathy and support.
Do NOT repeat the user's message.

User message:
{text}

Response:
""".strip()

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

        t = text_embedding(user_text)
        f = face_embedding(img)
        _ = gated_fusion(t, f)

        response = llm(build_prompt(user_text), max_new_tokens=200)[0]["generated_text"]

        st.subheader("Response")
        st.write(response)
