import streamlit as st
import numpy as np
from gradio_client import Client, handle_file
import tempfile
import requests

# -----------------------
# PAGE CONFIG
# -----------------------

st.set_page_config(page_title="MEPE", layout="wide")

st.markdown("""
<style>
.big-button button {
    width: 100%;
    height: 60px;
    font-size: 20px;
    font-weight: 600;
    border-radius: 12px;
    background-color: #4F46E5;
    color: white;
}

.persona-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #111827;
    border: 1px solid #374151;
}

.response-box {
    padding: 25px;
    border-radius: 14px;
    background-color: #0F172A;
    border: 1px solid #334155;
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# CONFIG
# -----------------------

HF_SPACE_ID = "upendrareddy1/mepe"
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

hf_client = Client(HF_SPACE_ID)

# -----------------------
# CALL HF SPACE
# -----------------------

def get_persona_embedding(text, image_bytes):

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(image_bytes)
            temp_path = tmp.name

        result = hf_client.predict(
            text=text,
            image=handle_file(temp_path),
            api_name="/mepe_inference"
        )

        persona_vector = result["persona_embedding"]

        return persona_vector, None

    except Exception as e:
        return None, str(e)

# -----------------------
# PERSONA INTERPRETER (LLM)
# -----------------------

def interpret_persona(persona_vector):

    prompt = f"""
You are a behavioral AI analyst.

You are given a 512-dimensional fused multimodal embedding
derived from text + facial emotion signals.

Embedding:
{persona_vector}

Interpret this embedding and summarize the persona in 3-4 lines.
Focus on:
- Communication style
- Emotional tone
- Energy level

Keep it simple and clear.
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a psychological embedding interpreter."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.4,
        "max_tokens": 250
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        return "Persona interpretation failed."

    result = response.json()
    return result["choices"][0]["message"]["content"]

# -----------------------
# RESPONSE GENERATOR (LLM)
# -----------------------

def generate_response(persona_vector, persona_summary, user_text):

    prompt = f"""
You are an emotionally intelligent assistant.

Detected Persona:
{persona_summary}

Full Multimodal Embedding:
{persona_vector}

User Message:
{user_text}

Generate a response aligned with the detected persona.
Adapt tone and emotional intensity accordingly.
Be natural and human.
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You generate persona-aligned responses."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        return f"LLM Error {response.status_code}: {response.text}"

    result = response.json()
    return result["choices"][0]["message"]["content"]

# -----------------------
# UI
# -----------------------

st.title("🧠 MEPE – Multimodal Emotion Persona Engine")

# INPUT ROW
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Input Signals")
    text_input = st.text_area("Message", height=150)

with col2:
    st.subheader("📷 Face Input")
    image_input = st.file_uploader(
        "Upload face image",
        type=["png", "jpg", "jpeg"]
    )

st.markdown("<br>", unsafe_allow_html=True)

st.markdown('<div class="big-button">', unsafe_allow_html=True)
generate = st.button("🚀 Generate Persona-Aware Response")
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# EXECUTION
# -----------------------

if generate:

    if not text_input or not image_input:
        st.error("Both text and image required.")
    else:

        with st.spinner("Extracting multimodal embedding..."):

            image_bytes = image_input.read()
            persona_vector, error = get_persona_embedding(text_input, image_bytes)

        if error:
            st.error(error)

        else:

            with st.spinner("Interpreting persona..."):
                persona_summary = interpret_persona(persona_vector)

            st.markdown("### 🔍 Detected Persona")
            st.markdown(
                f"""
                <div class="persona-box">
                {persona_summary}
                </div>
                """,
                unsafe_allow_html=True
            )

            with st.spinner("Generating response..."):
                reply = generate_response(persona_vector, persona_summary, text_input)

            st.markdown("### 🤖 Emotion-Aware Response")
            st.markdown(
                f"""
                <div class="response-box">
                {reply}
                </div>
                """,
                unsafe_allow_html=True
            )
