# Multimodal Emotion Persona Engine (MEPE)

MEPE is an end-to-end multimodal AI system that understands a user’s emotional state and communication persona from text and facial expressions, fuses these signals into a unified representation, and generates emotion-aware, persona-aligned responses using a large language model.

This project demonstrates production-style AI system design combining NLP, Computer Vision, Multimodal Fusion, and Controlled Text Generation.

Key Skills Demonstrated (ATS-Optimized)

Natural Language Processing (NLP)

Computer Vision (CNNs)

Transformer Models (DistilBERT, T5)

Multimodal Representation Learning

Emotion Recognition

Persona Modeling

Gated Multimodal Fusion

Prompt Engineering

Controlled Text Generation

TensorFlow / Keras

Hugging Face Transformers

End-to-End ML System Design

System Architecture Overview
## 🧩 Architecture Diagram

The following diagram illustrates the end-to-end flow of the Multimodal Emotion Persona Engine (MEPE), from raw inputs to persona-aware response generation.

![MEPE Architecture Diagram](assets/architecture.png)

Project Structure
mepe/
├── phase1_text_emotion/        # Text emotion classification (Transformer)
├── phase2_face_emotion/        # Facial emotion recognition (CNN)
├── phase3_inference/           # Independent inference pipelines
├── phase4_fusion/              # Multimodal fusion (gated attention)
├── phase5_persona_llm/
│   └── phase5_persona_llm.ipynb  # 5A + 5B + 5C (design → control → demo)
└── README.md

Note: Phase 5 (Design, Persona Control, and Demo) is intentionally implemented in a single notebook to preserve end-to-end reasoning and reproducibility.

Phase Breakdown
Phase 1 – Text Emotion Modeling

Transformer-based emotion classification using DistilBERT

Fine-tuned on multi-label emotion datasets

Outputs dense text emotion embeddings

Phase 2 – Face Emotion Modeling

CNN-based facial emotion recognition (FER-2013)

Outputs facial emotion probabilities and embeddings

Phase 3 – Independent Inference

Standalone inference for text and face emotion models

Model persistence and reproducibility

Phase 4 – Multimodal Fusion

Gated Fusion Mechanism combines text and face embeddings

Learns modality importance dynamically

Produces a unified persona embedding

Phase 5 – Persona-Aware LLM Generation

Implemented in phase5_persona_llm.ipynb:

5A – Design: Persona schema, control variables, LLM selection

5B – Persona Control: Rule-based policy mapping persona traits → behavioral controls

5C – Demo: End-to-end emotion-aware response generation

🔍 Demo (Research-Grade)
User Input (Text)

“I feel overwhelmed and frustrated with how things are going.”

Detected Persona (Multimodal)

Stress: Medium

Sadness: Medium

Emotional Intensity: Medium

Confidence: High

Formality Preference: Low

Derived Control Signals

Empathy Level: Medium

Response Style: Calm

Assertiveness: Medium

Formality: Casual

Generated Response

“It sounds like you’re dealing with a lot right now, and that can be genuinely exhausting. It might help to slow things down and focus on one small step at a time. If you’d like, we can think through what would make things feel more manageable.”

The response style is dynamically controlled by multimodal emotion signals and a persona-aware policy, not static prompting.

Why Gated Fusion (Design Decision)

Handles noisy or missing modalities

Computationally efficient compared to cross-attention

Interpretable modality weighting

Suitable for single-GPU environments (Colab / Kaggle)

This mirrors real-world system trade-offs in applied AI teams.

Real-World Applications

Emotion-aware conversational agents

Mental health and wellbeing assistants

Adaptive customer support systems

Human-centric AI interfaces

Personalized AI companions

Limitations & Future Work

Replace rule-based persona control with learned policy

Temporal modeling of emotion drift

Reinforcement learning for empathy optimization

Cross-attention fusion for long multimodal sequences

Real-time video-based inference

Reproducibility

All models, inference steps, and demos are runnable via the provided notebooks.
Pretrained models are loaded using Hugging Face–compatible formats.

Author Notes (Optional but Strong)

This project was built end-to-end by a single developer, emphasizing system-level thinking, engineering trade-offs, and research-inspired design over isolated model performance.
