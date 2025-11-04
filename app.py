import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from deep_translator import GoogleTranslator

# ------------------------------------------------
# ✨ Professional Modern Styling (No Background Image)
# ------------------------------------------------
def add_modern_style():
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top, #0b132b, #1c2541, #3a506b);
            color: #f5f6fa;
            font-family: 'Inter', sans-serif;
        }
        h1 {
            text-align: center;
            color: #e0e6ed;
            font-weight: 800;
            font-size: 2.5em;
            letter-spacing: 1px;
            text-shadow: 0 0 15px rgba(100, 181, 246, 0.6);
        }
        p, .stMarkdown {
            color: #dcdde1 !important;
            font-size: 17px !important;
            line-height: 1.6;
        }
        textarea {
            background: rgba(255,255,255,0.08);
            color: #f5f6fa !important;
            border-radius: 10px !important;
            border: 1px solid rgba(255,255,255,0.25) !important;
            font-size: 16px !important;
        }
        div.stButton > button {
            background: linear-gradient(90deg, #00a8cc, #007ea7);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.7em 1.5em;
            font-weight: bold;
            transition: 0.3s;
            width: 100%;
            font-size: 16px;
        }
        div.stButton > button:hover {
            background: linear-gradient(90deg, #007ea7, #00a8cc);
            transform: scale(1.03);
            box-shadow: 0 0 20px rgba(0,168,204,0.5);
        }
        .stAlert, .stSuccess, .stWarning, .stInfo {
            background-color: rgba(255,255,255,0.08) !important;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 10px;
            padding: 15px;
        }
        h2, h3 {
            color: #8bd3dd;
            margin-top: 20px;
        }
        hr {
            border: 1px solid rgba(255,255,255,0.2);
        }
        .stCaption {
            color: #9ca8b8 !important;
            font-size: 13px;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_modern_style()

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
# Streamlit UI
# ------------------------------------------------
st.title("🧠 Mind Lens 🔍")
st.markdown("<p style='text-align:center;'>Step in, let your words speak. Explore emotions, find balance, and connect with care wherever you are.</p>", unsafe_allow_html=True)

user_text = st.text_area("💬 Type or paste your text here:", height=150)

if st.button("✨ Analyze Now"):
    if not user_text.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        # Translate text to English if needed
        try:
            english_text = GoogleTranslator(source='auto', target='en').translate(user_text)
            st.info("🌍 Text translated to English (if needed).")
            st.markdown(f"**Translated text:** {english_text}")
        except Exception:
            english_text = user_text
            st.warning("⚠️ Translation service unavailable — using original text.")

        # Tokenize and predict
        inputs = tokenizer(english_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            pred_class = torch.argmax(outputs.logits, dim=1).item()

        label = label_mapping.get(pred_class, "Unknown")
        st.success(f"**🩺 Predicted Mental Health Category:** {label.upper()}")

        # Display helpful resources
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("💡 Helpful Suggestions & Resources:")
        for tip in resources.get(label, []):
            st.markdown(f"- {tip}")

        # Disclaimer
        st.markdown("<hr>", unsafe_allow_html=True)
        st.caption("⚠️ This tool is for informational support only and does not replace professional mental health advice.")
        st.caption("⚠️ Translations may not be perfect; always seek local professional help when needed.")