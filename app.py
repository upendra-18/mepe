import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from transformers import AutoTokenizer, TFDistilBertModel

# ================================
# STREAMLIT CONFIG
# ================================
st.set_page_config(page_title="MEPE", layout="centered")
st.title("🧠 MEPE – Multimodal Emotion Persona Engine")

# ================================
# LOAD MODELS (ONCE)
# ================================
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(
        "upendrareddy1/mepe-text-emotion"
    )
    text_encoder = TFDistilBertModel.from_pretrained(
        "upendrareddy1/mepe-text-emotion"
    )
    text_encoder.trainable = False

    face_model = tf.keras.models.load_model(
        "models/face_emotion/model.keras",
        compile=False
    )

    return tokenizer, text_encoder, face_model

tokenizer, text_encoder, face_model = load_models()

# ================================
# TEXT EMOTION
# ================================
def text_emotion(text):
    tokens = tokenizer(
        text,
        return_tensors="tf",
        truncation=True,
        padding=True,
        max_length=128
    )
    outputs = text_encoder(**tokens)
    emb = tf.reduce_mean(outputs.last_hidden_state, axis=1)
    return "neutral"  # replace with classifier if needed

# ================================
# FACE EMOTION
# ================================
def face_emotion(img):
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = face_model.predict(arr, verbose=0)[0]

    emotions = ["angry","disgust","fear","happy","sad","surprise","neutral"]
    return emotions[int(np.argmax(preds))]

# ================================
# MEPE LOGIC
# ================================
def resolve(text_e, face_e):
    return text_e if text_e != "neutral" else face_e

def generate_response(text, emotion):
    return f"I sense **{emotion}**. Talk to me about what's on your mind."

# ================================
# UI
# ================================
user_text = st.text_area("How are you feeling?")
image = st.camera_input("Capture your face")

if st.button("Analyze") and user_text and image:
    img = Image.open(image)

    with st.spinner("Understanding you..."):
        te = text_emotion(user_text)
        fe = face_emotion(img)
        final = resolve(te, fe)
        reply = generate_response(user_text, final)

    st.subheader("🧭 Emotion")
    st.write(final)

    st.subheader("💬 Response")
    st.write(reply)
