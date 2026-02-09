import streamlit as st
from gradio_client import Client
from PIL import Image
import io
import base64

# ================================
# STREAMLIT CONFIG
# ================================
st.set_page_config(page_title="MEPE", layout="centered")
st.title("🧠 MEPE – Multimodal Emotion Persona Engine")

# ================================
# GRADIO CLIENTS
# ================================
text_client = Client("upendrareddy1/mepe-text-emotion-api")
face_client = Client("upendrareddy1/mepe-face-emotion-api")

# ================================
# MODEL CALLS (OUTPUT-AWARE)
# ================================
def predict_text_emotion(text: str):
    """
    Expected output format:
    [[label, confidence]]
    """
    result = text_client.predict(
        text,
        api_name="/predict"
    )

    label = result[0][0]
    confidence = result[0][1]

    return label, confidence


def predict_face_emotion(img: Image.Image):
    """
    Expected output format:
    [[label, confidence]]
    """
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    result = face_client.predict(
        f"data:image/jpeg;base64,{b64}",
        api_name="/predict"
    )

    label = result[0][0]
    confidence = result[0][1]

    return label, confidence


# ================================
# FUSION LOGIC
# ================================
def resolve_final_emotion(text_out, face_out):
    """
    Fusion Strategy (Explainable):
    1. If both modalities agree → accept emotion directly
    2. If they disagree:
       a) If text confidence is high → trust text
       b) If face confidence is significantly higher → override text
       c) Otherwise → default to text (language carries intent)
    """

    text_label, text_conf = text_out
    face_label, face_conf = face_out

    # 1️⃣ Agreement case
    if text_label == face_label:
        return text_label

    # 2️⃣ Disagreement cases

    # Strong text signal → trust text
    if text_conf >= 0.70:
        return text_label

    # Face much more confident than text → trust face
    if face_conf - text_conf >= 0.25:
        return face_label

    # Ambiguous → default to text (intent > expression)
    return text_label


def generate_response(user_text: str, emotion: str) -> str:
    return (
        f"I sense **{emotion}**. "
        "Pause for a moment, breathe slowly, "
        "and focus on one small thing you can control right now."
    )

# ================================
# UI
# ================================
user_text = st.text_area("How are you feeling?")
image = st.camera_input("Capture your facial expression")

if st.button("Analyze & Respond"):
    if not user_text or image is None:
        st.warning("Text and face input required")
    else:
        img = Image.open(image)

        with st.spinner("Understanding you..."):
            text_out = predict_text_emotion(user_text)
            face_out = predict_face_emotion(img)
            final_emotion = resolve_final_emotion(text_out, face_out)
            response = generate_response(user_text, final_emotion)

        st.subheader("🧭 Inferred Emotional State")
        st.write(final_emotion)

        st.subheader("📊 Model Signals")
        st.write({
            "text_emotion": text_out,
            "face_emotion": face_out
        })

        st.subheader("💬 MEPE Response")
        st.write(response)
