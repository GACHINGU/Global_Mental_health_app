import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from deep_translator import GoogleTranslator

# ------------------------------------------------
# 🌌 Background + Improved Styling
# ------------------------------------------------
def add_beautiful_style():
    st.markdown(
        f"""
        <style>
        /* Background setup */
        .stApp {{
            background-image: url("https://i.pinimg.com/1200x/de/47/a4/de47a494d95218de602e749aaf6c9e67.jpg");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.85);
            z-index: -1;
        }}

        /* Title glow */
        h1 {{
            color: #f1f2f6;
            text-align: center;
            text-shadow: 0 0 10px #4a69bd, 0 0 20px #6a89cc;
            font-family: 'Poppins', sans-serif;
            font-weight: 800;
        }}

        /* Subtext */
        p, .stMarkdown {{
            color: #dcdde1 !important;
            font-size: 17px !important;
            line-height: 1.6;
            font-family: 'Open Sans', sans-serif;
        }}

        /* Card look for text area and results */
        .stTextArea, .stAlert, .stSuccess, .stWarning, .stInfo {{
            background-color: rgba(255, 255, 255, 0.08) !important;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 12px;
            padding: 15px;
        }}

        textarea {{
            background-color: rgba(255,255,255,0.1) !important;
            color: white !important;
            border-radius: 10px !important;
            border: 1px solid rgba(255,255,255,0.3) !important;
            font-size: 16px !important;
        }}

        /* Button */
        div.stButton > button {{
            background: linear-gradient(90deg, #4a69bd, #6a89cc);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6em 1.5em;
            font-weight: bold;
            transition: 0.3s;
            width: 100%;
            font-size: 16px;
        }}
        div.stButton > button:hover {{
            background: linear-gradient(90deg, #6a89cc, #4a69bd);
            transform: scale(1.03);
            box-shadow: 0 0 15px #4a69bd;
        }}

        /* Subheaders */
        h2, h3 {{
            color: #82ccdd;
            text-shadow: 0 0 5px #3c6382;
            margin-top: 20px;
        }}

        /* Horizontal rule */
        hr {{
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}

        /* Footer captions */
        .stCaption {{
            color: #95a5a6 !important;
            font-size: 13px;
            text-align: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_beautiful_style()

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