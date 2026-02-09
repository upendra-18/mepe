import streamlit as st
import requests
import numpy as np
import base64
from PIL import Image
import io

# ===============================
# CONFIG
# ===============================

TEXT_API = "https://upendrareddy1-mepe-text-emotion-api.hf.space/run/predict"
FACE_API = "https://upendrareddy1-mepe-face-emotion-api.hf.space/run/predict"

TEXT_DIM = 768
FACE_DIM = 256
FUSION_DIM = 512

REQUEST_TIMEOUT = 25  # seconds

# ===============================
# UTILS
# ===============================

def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

def safe_post(url, payload):
    r = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()

# ===============================
# HF SPACE CALLS
# ===============================

def predict_text_emotion(text: str):
    payload = {"data": [text]}
    res = safe_post(TEXT_API, payload)

    # EXPECTED: [embedding, label]
    embedding = np.array(res["data"][0], dtype=np.float32)
    label = res["data"][1]

    if embedding.shape[0] != TEXT_DIM:
        raise ValueError("Text embedding dimension mismatch")

    return embedding, label

def predict_face_emotion(img: Image.Image):
    b64 = image_to_base64(img)
    payload = {"data": [f"data:image/jpeg;base64,{b64}"]}
    res = safe_post(FACE_API, payload)

    embedding = np.array(res["data"][0], dtype=np.float32)
    label = res["data"][1]

    if embedding.shape[0] != FACE_DIM:
        raise ValueError("Face embedding dimension mismatch")

    return embedding, label

# ===============================
# GATED FUSION (PURE NUMPY)
# ===============================

def gated_fusion(text_emb, face_emb):
    """
    g = sigmoid(Wg[t;f])
    fused = g*t + (1-g)*f
    """

    # Project to same dim
    Wt = np.random.randn(TEXT_DIM, FUSION_DIM) * 0.01
    Wf = np.random.randn(FACE_DIM, FUSION_DIM) * 0.01
    Wg = np.random.randn(FUSION_DIM * 2, FUSION_DIM) * 0.01

    t = text_emb @ Wt
    f = face_emb @ Wf

    concat = np.concatenate([t, f])
    g = 1 / (1 + np.exp(-(concat @ Wg)))

    fused = g * t + (1 - g) * f
    return fused

# ===============================
# RESPONSE LOGIC (MEPE CORE)
# ===============================

def generate_mepe_response(text, emotion):
    return (
        f"I sense **{emotion}** in how you’re expressing yourself.\n\n"
        f"You don’t have to suppress it. Want to talk through what triggered this?"
    )

# ===============================
# STREAMLIT UI
# ===============================

st.set_page_config(page_title="MEPE", layout="centered")
st.title("🧠 MEPE — Multimodal Emotion Persona Engine")

user_text = st.text_area("💬 Say something", height=120)

image = st.camera_input("📸 Face Input")

run = st.button("Analyze")

if run:
    if not user_text or image is None:
        st.error("Text AND face input are required.")
        st.stop()

    try:
        img = Image.open(image)

        with st.spinner("Understanding you..."):
            text_emb, text_label = predict_text_emotion(user_text)
            face_emb, face_label = predict_face_emotion(img)

            fused_emb = gated_fusion(text_emb, face_emb)

            # RULE: face overrides text if conflict
            final_emotion = face_label if face_label != text_label else text_label

            response = generate_mepe_response(user_text, final_emotion)

        st.subheader("🧭 Inferred Emotion")
        st.write(final_emotion)

        st.subheader("💬 MEPE Response")
        st.write(response)

    except Exception as e:
        st.error(f"MEPE failed: {e}")
