import streamlit as st
import numpy as np
from gradio_client import Client, handle_file
import tempfile
import requests

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(page_title="MEPE", layout="wide")

st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

/* Full width primary button */
div.stButton > button {
    width: 100%;
    height: 70px;
    font-size: 22px;
    font-weight: 700;
    border-radius: 14px;
    background: linear-gradient(90deg,#6366F1,#8B5CF6);
    color: white;
    border: none;
}

div.stButton > button:hover {
    background: linear-gradient(90deg,#4F46E5,#7C3AED);
}

/* Persona box */
.persona-box {
    padding: 28px;
    border-radius: 16px;
    background-color: #0F172A;
    border: 1px solid #1E293B;
    font-size: 18px;
}

/* Response box */
.response-box {
    padding: 35px;
    border-radius: 18px;
    background-color: #0B1120;
    border: 1px solid #1E293B;
    font-size: 18px;
    line-height: 1.7;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

HF_SPACE_ID = "upendrareddy1/mepe"
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

hf_client = Client(HF_SPACE_ID)

# --------------------------------------------------
# GET PERSONA EMBEDDING
# --------------------------------------------------

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

# --------------------------------------------------
# PERSONA INTERPRETER (LLM)
# --------------------------------------------------

def interpret_persona(persona_vector):

    prompt = f"""
You are a behavioral AI analyst.

This is a fused multimodal embedding (text + face):

{persona_vector}

Summarize clearly in 3 lines:
- Communication style
- Emotional tone
- Energy level
Keep it concise.
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You interpret behavioral embeddings."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.4,
        "max_tokens": 200
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

# --------------------------------------------------
# RESPONSE GENERATOR
# --------------------------------------------------

def generate_response(persona_vector, persona_summary, user_text):

    prompt = f"""
Detected Persona:
{persona_summary}

Embedding:
{persona_vector}

User Message:
{user_text}

Generate a response aligned with the persona.
Adapt tone and emotional intensity accordingly.
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "Generate persona-aligned responses."},
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

# --------------------------------------------------
# UI LAYOUT
# --------------------------------------------------

st.markdown("## 🧠 MEPE – Multimodal Emotion Persona Engine")

st.markdown("")

# 50 / 50 GRID (FULL WIDTH)
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-title">📝 Input Signals</div>', unsafe_allow_html=True)
    text_input = st.text_area("", height=220, placeholder="Enter your message...")

with col2:
    st.markdown('<div class="section-title">📷 Face Input</div>', unsafe_allow_html=True)
    image_input = st.file_uploader("", type=["png", "jpg", "jpeg"])

st.markdown("")

# FULL WIDTH BUTTON
generate = st.button("🚀 Generate Persona-Aware Response")

st.markdown("")

# --------------------------------------------------
# EXECUTION
# --------------------------------------------------

if generate:

    if not text_input or not image_input:
        st.error("Both text and image are required.")
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

            st.markdown("")

            with st.spinner("Generating response..."):
                reply = generate_response(persona_vector, persona_summary, text_input)

            st.markdown("### 🤖 Generated Response")
            st.markdown(
                f"""
                <div class="response-box">
                {reply}
                </div>
                """,
                unsafe_allow_html=True
            )
