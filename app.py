import streamlit as st
from gradio_client import Client, handle_file
from PIL import Image
import tempfile
import os

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
# TEXT EMOTION CALL
# ================================
def predict_text_emotion(text: str):
    """
    Gradio contract:
    /predict(input_ids, attention_mask)

    Backend handles tokenization internally.
    We only need to pass placeholders.
    Output: [[label, confidence]]
    """
    result = text_client.predict(
        input_ids={"text": text},
        attention_mask={"text": text},
        api_name="/predict"
    )

    label = result[0][0]
    confidence = result[0][1]
    return label, confidence


# ================================
# FACE EMOTION CALL
# ================================
def predict_face_emotion(img: Image.Image):
    """
    Gradio contract:
    /predict(image)

    Uses handle_file for upload.
    Output: [[label, confidence]]
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = face_client.predict(
            image=handle_file(tmp_path),
            api_name="/predict"
        )

        label = result[0][0]
        confidence = result[0][1]
        return label, confidence

    finally:
        os.remove(tmp_path)


# ================================
# FUSION LOGIC (CORE INTELLIGENCE)
# ================================
def resolve_final_emotion(text_out, face_out):
    """
    Confidence-aware multimodal fusion.

    Rules:
    1. Agreement → accept emotion
    2. High-confidence text → trust intent
    3. Stronger face signal → override text
    4. Otherwise → default to text
    """

    text_label, text_conf = text_out
    face_label, face_conf = face_out

    # 1️⃣ Agreement
    if text_label == face_label:
        return text_label

    # 2️⃣ Strong linguistic intent
    if text_conf >= 0.70:
        return text_label

    # 3️⃣ Facial dominance
    if face_conf - text_conf >= 0.25:
        return face_label

    # 4️⃣ Safe default
    return text_label


# ================================
# RESPONSE GENERATION (SIMPLE)
# ================================
def generate_response(user_text: str, emotion: str) -> str:
    return (
        f"I sense **{emotion}**. "
        "Pause for a moment, take a slow breath, "
        "and focus on one small thing you can control right now."
    )


# ================================
# UI
# ================================
user_text = st.text_area("How are you feeling?")
image = st.camera_input("Capture your facial expression")

if st.button("Analyze & Respond"):
    if not user_text or image is None:
        st.warning("Both text and face input are required.")
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
