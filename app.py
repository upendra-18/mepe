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
# Persona Summary Logic
# -----------------------

def summarize_persona(persona_vector):
    vec = np.array(persona_vector)

    intensity = float(np.linalg.norm(vec))
    positivity = float(np.mean(vec[:100]))
    dominance = float(np.mean(vec[100:200]))

    # Simple persona type heuristic
    if positivity > 0.05:
        persona_type = "Warm & Expressive"
    elif dominance > 0.05:
        persona_type = "Confident & Assertive"
    else:
        persona_type = "Analytical & Reserved"

    return {
        "emotional_intensity": round(intensity, 2),
        "positivity_score": round(positivity, 3),
        "dominance_score": round(dominance, 3),
        "persona_type": persona_type
    }

# -----------------------
# Call HF Space
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
# LLM Response Generator (Groq)
# -----------------------

def generate_response(persona_vector, user_text):

    prompt = f"""
You are an emotionally intelligent assistant.

Persona vector (512-dim latent embedding):
{persona_vector}

User message:
{user_text}

Generate a supportive, emotionally aligned response.
Keep it natural and human.
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
# Streamlit UI
# -----------------------

st.set_page_config(page_title="MEPE", layout="wide")

st.markdown("""
<style>
.big-title {
    font-size: 42px;
    font-weight: 800;
}
.subtle {
    color: #888;
}
.persona-card {
    padding: 20px;
    border-radius: 15px;
    background-color: #111;
}
.response-card {
    padding: 20px;
    border-radius: 15px;
    background-color: #1a1a1a;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🧠 MEPE</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Multimodal Emotion Persona Engine</div>', unsafe_allow_html=True)

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📝 Input Signals")
    text_input = st.text_area("Message")
    image_input = st.file_uploader("Face Image", type=["png", "jpg", "jpeg"])

with col2:
    st.markdown("### 🔍 Detected Persona")
    persona_placeholder = st.empty()

st.divider()

if st.button("Generate Emotion-Aware Response", use_container_width=True):

    if not text_input or not image_input:
        st.error("Both text and image required.")
    else:

        with st.spinner("Extracting multimodal persona representation..."):
            image_bytes = image_input.read()
            persona_vector, error = get_persona_embedding(text_input, image_bytes)

        if error:
            st.error(error)
        else:

            persona_info = summarize_persona(persona_vector)

            with persona_placeholder.container():
                st.markdown("#### 🎭 Persona Profile")
                st.markdown(f"**Type:** {persona_info['persona_type']}")
                st.markdown(f"**Embedding Dimension:** 512")
                st.metric("Emotional Intensity", persona_info["emotional_intensity"])
                st.progress(min(persona_info["emotional_intensity"] / 50, 1.0))

                with st.expander("Technical Details"):
                    st.write("Raw Embedding (first 10 dims):")
                    st.write(persona_vector[:10])

            with st.spinner("Generating persona-aligned response..."):
                reply = generate_response(persona_vector, text_input)

            st.divider()
            st.markdown("### 🤖 Emotion-Aware Response")

            st.markdown("""
            <div class="response-card">
            """, unsafe_allow_html=True)

            st.write(reply)

            st.markdown("</div>", unsafe_allow_html=True)

