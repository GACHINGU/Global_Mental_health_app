import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from deep_translator import GoogleTranslator

# ------------------------------------------------
# 🌤️ Modern Light Theme Styling
# ------------------------------------------------
def add_light_style():
    st.markdown(
        """
        <style>
        /* Base background */
        .stApp {
            background: linear-gradient(135deg, #f9fbff, #f4f8fc);
            color: #1c1c1e;
            font-family: 'Inter', sans-serif;
        }

        /* Title */
        h1 {
            text-align: center;
            color: #1d3557;
            font-weight: 800;
            font-size: 2.6em;
            letter-spacing: 0.5px;
        }

        /* Paragraphs */
        p, .stMarkdown {
            color: #2b2d42 !important;
            font-size: 17px !important;
            line-height: 1.6;
        }

        /* Text Area */
        textarea {
            background: #ffffff !important;
            color: #1c1c1e !important;
            border-radius: 10px !important;
            border: 1px solid #ced6e0 !important;
            font-size: 16px !important;
            box-shadow: 0px 2px 4px rgba(0,0,0,0.05);
        }

        /* Button */
        div.stButton > button {
            background: linear-gradient(90deg, #0077b6, #0096c7);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.7em 1.5em;
            font-weight: 600;
            transition: 0.3s;
            width: 100%;
            font-size: 16px;
            box-shadow: 0px 3px 6px rgba(0, 119, 182, 0.3);
        }
        div.stButton > button:hover {
            background: linear-gradient(90deg, #0096c7, #0077b6);
            transform: translateY(-2px);
            box-shadow: 0px 5px 12px rgba(0, 119, 182, 0.4);
        }

        /* Info/Alert Cards */
        .stAlert, .stSuccess, .stWarning, .stInfo {
            background-color: #ffffff !important;
            border: 1px solid #e1e8ef !important;
            border-radius: 12px !important;
            padding: 15px !important;
            box-shadow: 0px 4px 8px rgba(0,0,0,0.05);
        }

        /* Subheaders */
        h2, h3 {
            color: #005f73;
            margin-top: 25px;
        }

        /* Horizontal Rule */
        hr {
            border: 1px solid #e0e6ed;
        }

        /* Captions / Footers */
        .stCaption {
            color: #6c757d !important;
            font-size: 13px;
            text-align: center;
        }

        /* Center align intro text */
        .center-text {
            text-align: center;
            color: #3a3a3a;
            font-size: 16.5px;
            margin-bottom: 25px;
        }

        /* Smooth transitions */
        * {
            transition: all 0.25s ease-in-out;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_light_style()

# ------------------------------------------------
# Load model and tokenizer
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
st.markdown("<p class='center-text'>Step in, let your words speak. Explore emotions, find balance, and connect with care wherever you are.</p>", unsafe_allow_html=True)

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