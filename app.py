# ================================
# MEPE Streamlit App (FINAL)
# ================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import numpy as np
from PIL import Image

import tensorflow as tf
import keras
from transformers import AutoTokenizer, TFDistilBertModel, pipeline

# -------------------------------
# Streamlit config
# -------------------------------
st.set_page_config(page_title="MEPE Demo", layout="centered")
st.title("🧠 MEPE – Multimodal Emotion Persona Engine")

# -------------------------------
# Load models (cached)
# -------------------------------
@st.cache_resource
def load_models():
    from transformers import AutoTokenizer, TFDistilBertModel, pipeline
    from huggingface_hub import hf_hub_download
    import keras

    # ---------- TEXT MODEL ----------
    tokenizer = AutoTokenizer.from_pretrained(
        "upendrareddy1/mepe-text-emotion"
    )

    text_encoder = TFDistilBertModel.from_pretrained(
        "upendrareddy1/mepe-text-emotion"
    )
    text_encoder.trainable = False

    # ---------- FACE MODEL (FIX) ----------
    face_model_path = hf_hub_download(
        repo_id="upendrareddy1/face-emotion-keras",
        filename="model.keras"
    )

    face_model = keras.models.load_model(
        face_model_path,
        compile=False
    )

    # ---------- LLM ----------
    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=-1
    )

    return tokenizer, text_encoder, face_model, llm



tokenizer, text_encoder, face_model, llm = load_models()

# -------------------------------
# Inference helpers
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
    return emb.numpy()[0]


def face_embedding(img: Image.Image) -> np.ndarray:
    img = img.convert("L").resize((48, 48))
    arr = np.array(img, dtype="float32") / 255.0
    arr = arr.reshape(1, 48, 48, 1)
    return face_model.predict(arr, verbose=0)[0]


def gated_fusion(t: np.ndarray, f: np.ndarray) -> np.ndarray:
    alpha = np.mean(t) / (np.mean(t) + np.mean(f) + 1e-6)
    return alpha * t + (1.0 - alpha) * f


def persona_control():
    return {
        "stress": "medium",
        "empathy": "medium",
        "tone": "calm",
        "formality": "casual"
    }


def build_prompt(user_text: str, persona: dict) -> str:
    return f"""
You are an emotionally intelligent assistant.
Respond with empathy and support.
Do NOT repeat the user's message.

Context:
- Stress level: {persona['stress']}
- Empathy needed: {persona['empathy']}
- Tone: {persona['tone']}
- Formality: {persona['formality']}

User message:
{user_text}

Response:
""".strip()

# -------------------------------
# UI
# -------------------------------
user_text = st.text_area("How are you feeling?")
image = st.camera_input("Capture your facial expression")

if st.button("Analyze & Respond"):
    if not user_text or image is None:
        st.warning("Both text input and camera capture are required.")
    else:
        img = Image.open(image)

        t_emb = text_embedding(user_text)
        f_emb = face_embedding(img)
        fused = gated_fusion(t_emb, f_emb)

        persona = persona_control()
        prompt = build_prompt(user_text, persona)

        response = llm(prompt, max_new_tokens=200)[0]["generated_text"]

        st.subheader("Detected Persona")
        st.json(persona)

        st.subheader("Response")
        st.write(response)
