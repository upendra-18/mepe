import streamlit as st
import numpy as np
from gradio_client import Client, handle_file
import tempfile
import requests

# -----------------------
# CONFIG
# -----------------------

HF_SPACE_ID = "upendrareddy1/mepe"
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

hf_client = Client(HF_SPACE_ID)

# -----------------------
# Persona Embedding
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
        return np.array(persona_vector), None

    except Exception as e:
        return None, str(e)

# -----------------------
# Persona Interpretation (LIGHT MODEL)
# -----------------------

def interpret_persona(persona_vector):

    mean_val = float(np.mean(persona_vector))
    std_val = float(np.std(persona_vector))
    norm_val = float(np.linalg.norm(persona_vector))
    max_val = float(np.max(persona_vector))
    min_val = float(np.min(persona_vector))

    summary = f"""
Embedding statistics:
Mean: {mean_val}
Std: {std_val}
L2 Norm: {norm_val}
Max Activation: {max_val}
Min Activation: {min_val}

Infer:
- Communication style
- Emotional stability
- Confidence level
- Social energy
Keep it short and structured.
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-8b-instant",   # smaller model
        "messages": [
            {"role": "system", "content": "You analyze psychological embedding summaries."},
            {"role": "user", "content": summary}
        ],
        "temperature": 0.3,
        "max_tokens": 200
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload
    )

    if response.status_code != 200:
        return "Persona interpretation unavailable."

    result = response.json()
    return result["choices"][0]["message"]["content"]

# -----------------------
# Emotion-Aware Response (STRONG MODEL)
# -----------------------

def generate_response(persona_vector, user_text):

    prompt = f"""
You are an emotionally intelligent assistant.

User message:
{user_text}

Use the latent persona signal (embedding norm={np.linalg.norm(persona_vector):.2f}) 
to subtly adapt tone and depth.

Generate a supportive, emotionally aligned response.
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are emotionally intelligent."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 400
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload
    )

    if response.status_code != 200:
        return f"LLM Error {response.status_code}: {response.text}"

    result = response.json()
    return result["choices"][0]["message"]["content"]

# -----------------------
# UI
# -----------------------

st.set_page_config(page_title="MEPE", layout="centered")

st.markdown("""
<style>
.stButton > button {
    height: 60px;
    font-size: 18px;
    font-weight: 600;
    border-radius: 15px;
    background: linear-gradient(90deg, #6C63FF, #00D4FF);
    color: white;
    border: none;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 MEPE")
st.caption("Multimodal Emotion Persona Engine")

st.markdown("### 📝 Input Signals")
text_input = st.text_area("Message")
image_input = st.file_uploader("Face Image", type=["png", "jpg", "jpeg"])

generate = st.button("🚀 Analyze Persona & Generate Response", use_container_width=True)

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

            st.markdown("## 🔍 Detected Persona")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Embedding Dimension", "512")
                st.metric("Vector Norm", round(np.linalg.norm(persona_vector), 2))

            with col2:
                st.metric("Mean Activation", round(np.mean(persona_vector), 4))
                st.metric("Std Deviation", round(np.std(persona_vector), 4))

            with st.spinner("Interpreting persona traits..."):
                persona_summary = interpret_persona(persona_vector)

            st.markdown("### 🎭 Behavioral Interpretation")
            st.info(persona_summary)

            st.markdown("---")

            with st.spinner("Generating emotion-aware response..."):
                reply = generate_response(persona_vector, text_input)

            st.markdown("## 🤖 Emotion-Aware Response")
            st.success(reply)

            with st.expander("Technical View (Embedding Sample)"):
                st.write(persona_vector[:20])
