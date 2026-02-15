import streamlit as st
import numpy as np
from gradio_client import Client, handle_file
import tempfile
import requests

# =====================================================
# CONFIG
# =====================================================

HF_SPACE_ID = "upendrareddy1/mepe"
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

hf_client = Client(HF_SPACE_ID)

GROQ_MODEL = "llama-3.3-70b-versatile"

# =====================================================
# MULTIMODAL EMBEDDING FETCH
# =====================================================

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

# =====================================================
# EMBEDDING SIGNAL EXTRACTION
# =====================================================

def extract_embedding_signals(vec):
    vec = np.array(vec)

    return {
        "dimension": len(vec),
        "norm": float(np.linalg.norm(vec)),
        "mean": float(np.mean(vec)),
        "std": float(np.std(vec)),
        "max_activation": float(np.max(vec)),
        "min_activation": float(np.min(vec))
    }

# =====================================================
# GROQ LLM CALL
# =====================================================

def call_groq(messages):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
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

# =====================================================
# PERSONA INTERPRETATION (EMBEDDING-CONDITIONED)
# =====================================================

def interpret_persona(signals, user_text):

    prompt = f"""
You are analyzing a multimodal personality embedding.

The embedding was generated from:
- Text semantic encoding (Transformer)
- Facial emotion encoding (CNN)
- Learned gated fusion network

Embedding statistics:
Dimension: {signals['dimension']}
Norm: {signals['norm']:.3f}
Mean: {signals['mean']:.4f}
Std Deviation: {signals['std']:.4f}
Max Activation: {signals['max_activation']:.3f}
Min Activation: {signals['min_activation']:.3f}

User message:
"{user_text}"

Based on both embedding signals and message content,
infer:

1. Communication Tone
2. Emotional Intensity
3. Confidence Level
4. Social Energy

Respond in short bullet format.
Be concise and professional.
"""

    messages = [
        {"role": "system", "content": "You are a psychological signal interpreter."},
        {"role": "user", "content": prompt}
    ]

    return call_groq(messages)

# =====================================================
# RESPONSE GENERATION (CONDITIONED ON PERSONA + EMBEDDING)
# =====================================================

def generate_response(persona_summary, signals, user_text):

    prompt = f"""
You are an emotionally intelligent assistant.

Multimodal Persona Summary:
{persona_summary}

Embedding Signals:
Norm: {signals['norm']:.3f}
Std: {signals['std']:.3f}

User message:
"{user_text}"

Generate a response that is aligned to:
- Communication tone
- Emotional intensity
- Confidence level
- Social energy

The response must feel personalized and adaptive.
Do not mention embeddings.
Keep it human and natural.
"""

    messages = [
        {"role": "system", "content": "You generate emotionally adaptive responses."},
        {"role": "user", "content": prompt}
    ]

    return call_groq(messages)

# =====================================================
# STREAMLIT UI
# =====================================================

st.set_page_config(page_title="MEPE", layout="centered")

st.markdown("""
# 🧠 MEPE  
### Multimodal Emotion Persona Engine  
Fusing text + facial emotion → latent persona → adaptive response
""")

st.divider()

st.subheader("📝 Input Signals")

text_input = st.text_area("Message")
image_input = st.file_uploader("Face Image", type=["png", "jpg", "jpeg"])

st.divider()

if st.button("🚀 Generate Adaptive Response", use_container_width=True):

    if not text_input or not image_input:
        st.error("Both text and image required.")
    else:

        with st.spinner("Analyzing multimodal signals..."):

            image_bytes = image_input.read()
            persona_vector, error = get_persona_embedding(text_input, image_bytes)

        if error:
            st.error(error)
        else:

            signals = extract_embedding_signals(persona_vector)

            with st.spinner("Interpreting persona from embedding..."):
                persona_summary = interpret_persona(signals, text_input)

            st.divider()
            st.subheader("🎭 Persona Snapshot")
            st.write(persona_summary)

            with st.spinner("Generating emotionally aligned response..."):
                reply = generate_response(persona_summary, signals, text_input)

            st.divider()
            st.subheader("🤖 Emotion-Aware Response")
            st.write(reply)
