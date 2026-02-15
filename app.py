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

# Initialize HF Space client
hf_client = Client(HF_SPACE_ID)


# -----------------------
# Call HF Space (Fusion Model)
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
# Embedding → Interpretable Persona
# -----------------------

def interpret_embedding(vec):
    vec = np.array(vec)

    norm = np.linalg.norm(vec)
    mean = np.mean(vec)
    std = np.std(vec)

    # Simple interpretable heuristics (NOT fake psychology)
    if norm > 11:
        energy = "High"
    elif norm > 9:
        energy = "Moderate"
    else:
        energy = "Calm"

    if std > 0.55:
        expressiveness = "Expressive"
    elif std > 0.4:
        expressiveness = "Balanced"
    else:
        expressiveness = "Reserved"

    if mean > 0.05:
        tone = "Externally Oriented"
    elif mean < -0.05:
        tone = "Internally Reflective"
    else:
        tone = "Neutral"

    return {
        "energy": energy,
        "expressiveness": expressiveness,
        "tone": tone,
        "norm": round(norm, 2)
    }


# -----------------------
# LLM Response Generator (Groq)
# -----------------------

def generate_response(persona_vector, user_text):

    prompt = f"""
You are an emotionally intelligent assistant.

Persona embedding (512-dim fused multimodal signal):
{persona_vector}

User message:
{user_text}

Generate a supportive, emotionally aligned response.
Keep it natural, grounded, and insightful.
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
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        return f"LLM Error {response.status_code}: {response.text}"

    result = response.json()
    return result["choices"][0]["message"]["content"]


# -----------------------
# UI Styling
# -----------------------

st.set_page_config(page_title="MEPE", layout="centered")

st.markdown("""
<style>
.big-button > button {
    background-color: #6C63FF;
    color: white;
    font-size: 18px;
    padding: 0.6em 1.2em;
    border-radius: 12px;
    border: none;
}
.big-button > button:hover {
    background-color: #5146D8;
}
.persona-box {
    padding: 15px;
    border-radius: 12px;
    background-color: #111827;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# -----------------------
# Layout
# -----------------------

st.title("🧠 MEPE")
st.subheader("Multimodal Emotion Persona Engine")

st.markdown("### 📝 Input Signals")

text_input = st.text_area("Message")
image_input = st.file_uploader("Face Image", type=["png", "jpg", "jpeg"])

generate_btn = st.container()
with generate_btn:
    generate = st.button("🚀 Analyze & Generate Response", key="generate", help="Run multimodal fusion + response generation")

if generate:

    if not text_input or not image_input:
        st.error("Both text and image are required.")
    else:

        with st.spinner("Running multimodal fusion..."):
            image_bytes = image_input.read()
            persona_vector, error = get_persona_embedding(text_input, image_bytes)

        if error:
            st.error(error)

        else:

            interpretation = interpret_embedding(persona_vector)

            st.markdown("## 🔍 Detected Persona")

            st.markdown(f"""
            <div class="persona-box">
            <h4>🎭 Persona Snapshot</h4>
            <b>Communication Tone:</b> {interpretation['tone']}<br>
            <b>Expressiveness:</b> {interpretation['expressiveness']}<br>
            <b>Energy Signal:</b> {interpretation['energy']}<br><br>
            <small>Derived from fused 512-dim multimodal embedding (text + face). Probabilistic signal interpretation.</small>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("## 🤖 Emotion-Aware Response")

            with st.spinner("Generating aligned response..."):
                reply = generate_response(persona_vector, text_input)

            st.success("Response generated.")
            st.write(reply)
