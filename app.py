import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from deep_translator import GoogleTranslator

# ------------------------------------------------
# Load model and tokenizer directly from Hugging Face
# ------------------------------------------------
@st.cache_resource
def load_model_and_tokenizer():
    repo_id = "Legend092/roberta-mentalhealth"
    model = RobertaForSequenceClassification.from_pretrained(repo_id)
    tokenizer = RobertaTokenizer.from_pretrained(repo_id)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# ------------------------------------------------
# Label Mapping
# ------------------------------------------------
label_mapping = {
    0: "anxiety",
    1: "bipolar",
    2: "depression",
    3: "normal",
    4: "personality disorder",
    5: "stress",
    6: "suicidal"
}

# ------------------------------------------------
# Helpful Resources
# ------------------------------------------------
resources = {
    "anxiety": [
        "Try slow breathing: inhale 4s, hold 4s, exhale 6s.",
        "Visit: [https://www.anxietycentre.com](https://www.anxietycentre.com)",
        "Talk to a trusted friend or counselor."
    ],
    "depression": [
        "You’re not alone — reaching out helps more than you think.",
        "Call your local helpline or message a friend.",
        "Resource: [https://findahelpline.com](https://findahelpline.com)"
    ],
    "stress": [
        "Take a short walk or stretch for 5 minutes.",
        "Practice deep breathing or listen to calm music.",
        "Resource: [https://www.stress.org](https://www.stress.org)"
    ],
    "bipolar": [
        "Track your mood daily to notice patterns.",
        "Keep routines consistent — especially sleep.",
        "Learn more: [https://www.nami.org](https://www.nami.org)"
    ],
    "personality disorder": [
        "Connecting with a therapist can really help you understand yourself.",
        "Try journaling to track emotions and triggers.",
        "Info: [https://www.mind.org.uk](https://www.mind.org.uk)"
    ],
    "suicidal": [
        "If you feel unsafe, **please reach out now**.",
        "Find help worldwide: [https://findahelpline.com](https://findahelpline.com)",
        "In Kenya: Befrienders Kenya – 0722 178177",
        "In the US: 988 Suicide & Crisis Lifeline"
    ],
    "normal": [
        "You seem balanced right now — keep practicing healthy habits!",
        "Maintain connections and regular breaks for mental wellness."
    ]
}

# ------------------------------------------------
# 🌙 Enhanced Dark Theme + Animated Header + Glow
# ------------------------------------------------
st.markdown("""
    <style>
    /* Base dark background */
    .stApp {
        background-color: #0b0f16;
        color: #e8eaed;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Animated gradient header */
    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    h1 {
        text-align: center;
        font-weight: 700;
        font-size: 2.6em;
        background: linear-gradient(270deg, #00bcd4, #2196f3, #00bcd4);
        background-size: 600% 600%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientMove 6s ease infinite;
    }

    /* Text area with glowing border */
    textarea {
        background-color: #1b1f27 !important;
        color: #e8eaed !important;
        border-radius: 12px !important;
        border: 1px solid #30343b !important;
        box-shadow: 0 0 10px rgba(0, 188, 212, 0.25);
        transition: box-shadow 0.3s ease-in-out;
    }
    textarea:focus {
        box-shadow: 0 0 18px rgba(33, 150, 243, 0.45);
        outline: none !important;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #00bcd4, #2196f3);
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: bold;
        transition: 0.3s;
        width: 100%;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #2196f3, #00bcd4);
        transform: scale(1.03);
        box-shadow: 0 0 10px rgba(0, 188, 212, 0.5);
    }

    /* Section Headers */
    h2, h3, h4 {
        color: #00bcd4;
        font-weight: 600;
    }

    /* Links */
    a {
        color: #4fc3f7 !important;
        text-decoration: none !important;
    }
    a:hover {
        text-decoration: underline !important;
    }

    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Streamlit UI
# ------------------------------------------------
st.title("🧠 Mind Lens")

st.markdown(
    "<p style='text-align:center; color:#9aa0a6; font-size:17px;'>"
    "Step in, let your words speak, explore emotions, find balance, and connect with care wherever you are."
    "</p>",
    unsafe_allow_html=True
)

user_text = st.text_area("💬 Type or paste your text here:", height=150)

if st.button("🔍 Analyze"):
    if not user_text.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        try:
            english_text = GoogleTranslator(source='auto', target='en').translate(user_text)
            st.info("🌍 Text has been translated to English (if needed).")
            st.markdown(f"**Translated text:** {english_text}")
        except Exception:
            english_text = user_text
            st.warning("⚠️ Translation service unavailable — using original text.")

        inputs = tokenizer(english_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            pred_class = torch.argmax(outputs.logits, dim=1).item()

        label = label_mapping.get(pred_class, "Unknown")
        st.success(f"**Predicted Mental Health Category:** {label.upper()}")

        st.markdown("---")
        st.subheader("💬 Helpful Suggestions & Resources:")
        for tip in resources.get(label, []):
            st.markdown(f"- {tip}")

        st.markdown("---")
        st.caption("⚠️ This tool is for informational support only and does not replace professional mental health advice.")
        st.caption("Disclaimer⚠️: Translations may not be perfect; always seek local professional help when needed.")
