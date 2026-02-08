import streamlit as st
import requests
import time
import base64
import io
from PIL import Image

# ================================
# CONFIG
# ================================
TEXT_SPACE = "https://upendrareddy1-mepe-text-emotion-api.hf.space"
FACE_SPACE = "https://upendrareddy1-mepe-face-emotion-api.hf.space"

st.set_page_config(page_title="MEPE", layout="centered")
st.title("🧠 MEPE – Multimodal Emotion Persona Engine")

# ================================
# GRADIO QUEUE CALL (CORRECT WAY)
# ================================
def gradio_predict(space_url, data, timeout=30):
    # Step 1: submit job
    r = requests.post(
        f"{space_url}/gradio_api/call/predict",
        json={"data": data},
        timeout=timeout
    )
    r.raise_for_status()
    event_id = r.json()["event_id"]

    # Step 2: poll result
    start = time.time()
    while True:
        if time.time() - start > timeout:
            raise RuntimeError("Gradio call timed out")

        r = requests.get(
            f"{space_url}/gradio_api/call/predict/{event_id}",
            timeout=timeout
        )

        for line in r.text.splitlines():
            if line.startswith("data:"):
                payload = line.replace("data:", "").strip()
                if payload and payload != "null":
                    return eval(payload)  # gradio returns python-like list

        time.sleep(0.5)

# ================================
# MODEL CALLS
# ================================
def predict_text_emotion(text):
    result = gradio_predict(TEXT_SPACE, [text])
    return result[0][0]   # label

def predict_face_emotion(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    result = gradio_predict(FACE_SPACE, [f"data:image/jpeg;base64,{b64}"])
    return result[0][0]   # label

def resolve_final_emotion(text_e, face_e):
    if text_e == face_e:
        return text_e
    return text_e  # text has priority (as you defined earlier)

def generate_response(user_text, emotion):
    return f"I sense **{emotion}**. Take a moment, breathe, and focus on one small thing you can control right now."

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
            text_e = predict_text_emotion(user_text)
            face_e = predict_face_emotion(img)
            final_e = resolve_final_emotion(text_e, face_e)
            response = generate_response(user_text, final_e)

        st.subheader("🧭 Inferred Emotional State")
        st.write(final_e)

        st.subheader("💬 MEPE Response")
        st.write(response)
