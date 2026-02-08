import time
import base64
import io
import requests
import streamlit as st
from PIL import Image

# ================================
# CONFIG
# ================================
TEXT_API = "https://upendrareddy1-mepe-text-emotion-api.hf.space"
FACE_API = "https://upendrareddy1-mepe-face-emotion-api.hf.space"

# ================================
# GRADIO CALL HELPERS
# ================================
def gradio_predict(api_base: str, data: list):
    # enqueue
    r = requests.post(
        f"{api_base}/gradio_api/call/predict",
        json={"data": data},
        timeout=60
    )
    r.raise_for_status()
    event_id = r.json()["event_id"]

    # poll
    while True:
        r = requests.get(
            f"{api_base}/gradio_api/call/predict/{event_id}",
            timeout=60
        )
        r.raise_for_status()
        result = r.json()

        if result["status"] == "completed":
            return result["data"]

        time.sleep(0.5)

# ================================
# MODEL CALLS
# ================================
def predict_text_emotion(text: str) -> str:
    result = gradio_predict(TEXT_API, [text])
    return result[0][0]

def predict_face_emotion(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    result = gradio_predict(
        FACE_API,
        [f"data:image/jpeg;base64,{b64}"]
    )
    return result[0][0]

# ================================
# FUSION LOGIC (SIMPLE, CORRECT)
# ================================
def resolve_final_emotion(text_e: str, face_e: str) -> str:
    if text_e == face_e:
        return text_e
    return f"{text_e} (text) / {face_e} (face)"

# ================================
# RESPONSE GENERATION (MEPE CORE)
# ================================
def generate_mepe_response(user_text: str, emotion: str) -> str:
    return (
        f"I sense **{emotion}**.\n\n"
        f"Based on what you said: *{user_text}*, "
        f"take a moment to breathe and reflect. "
        f"Would you like to talk more about it?"
    )

# ================================
# STREAMLIT UI
# ================================
st.set_page_config("MEPE", layout="centered")
st.title("🧠 MEPE – Emotion-Aware AI")

user_text = st.text_area("How are you feeling?")
image = st.camera_input("Capture your facial expression")

if st.button("Analyze"):
    if not user_text or not image:
        st.warning("Both text and image are required.")
    else:
        img = Image.open(image)

        with st.spinner("Understanding you..."):
            text_e = predict_text_emotion(user_text)
            face_e = predict_face_emotion(img)
            final_e = resolve_final_emotion(text_e, face_e)
            response = generate_mepe_response(user_text, final_e)

        st.subheader("🧭 Inferred Emotion")
        st.write(final_e)

        st.subheader("💬 MEPE Response")
        st.write(response)
