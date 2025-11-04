import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from deep_translator import GoogleTranslator

# ------------------------------------------------
# ✨ Elegant Centered Layout + Professional Theme
# ------------------------------------------------
def add_elegant_style():
    st.markdown(
        """
        <style>
        /* Global layout */
        .stApp {
            background: linear-gradient(135deg, #f7faff, #e8f1f8);
            font-family: 'Inter', sans-serif;
            color: #1c1c1e;
        }

        /* Header bar */
        .main-header {
            text-align: center;
            background: linear-gradient(90deg, #0077b6, #0096c7);
            color: white;
            padding: 1.2rem;
            border-radius: 0 0 20px 20px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: 0.5px;
        }

        /* Centered content box */
        .content-box {
            background: #ffffff;
            border-radius: 20px;
            padding: 2.5rem;
            box-shadow: 0px 6px 16px rgba(0,0,0,0.08);
            width: 75%;
            margin: 2rem auto;
            max-width: 800px;
        }

        /* Buttons */
        div.stButton > button {
            background: linear-gradient(90deg, #0096c7, #00b4d8);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.8em 1.6em;
            font-weight: 600;
            font-size: 16px;
            width: 100%;
            transition: 0.3s;
            box-shadow: 0px 3px 6px rgba(0, 150, 199, 0.3);
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0px 5px 12px rgba(0, 150, 199, 0.4);
            background: linear-gradient(90deg, #00b4d8, #0096c7);
        }

        /* Text area */
        textarea {
            background: #fdfefe !important;
            border-radius: 10px !important;
            border: 1px solid #d7e3eb !important;
            font-size: 16px !important;
            box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
        }

        /* Cards (info, warning, etc.) */
        .stAlert, .stSuccess, .stWarning, .stInfo {
            background-color: #ffffff !important;
            border: 1px solid #e6edf3 !important;
            border-radius: 12px !important;
            box-shadow: 0px 3px 8px rgba(0,0,0,0.05);
            padding: 15px !important;
        }

        /* Subtitles and sections */
        h2, h3 {
            color: #0077b6;
            margin-top: 25px;
        }

        /* Description text */
        .intro-text {
            text-align: center;
            font-size: 17px;
            color: #495057;
            margin-bottom: 1.5rem;
        }

        /* Footer */
        .footer {
            text-align: center;
            margin-top: 3rem;
            font-size: 14px;
            color: #6c757d;
            padding: 1rem 0;
        }
        .footer a {
            color: #0096c7;
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }

        /* Smooth transitions */
        * {
            transition: all 0.25s ease-in-out;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_elegant_style()

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
st.markdown("<div class='main-header'>🧠 Mind Lens</div>", unsafe_allow_html=True)

st.markdown("<div class='content-box'>", unsafe_allow_html=True)

st.markdown("<p class='intro-text'>Let your words reveal your emotional tone. Mind Lens helps you explore, understand, and find balance with care and empathy.</p>", unsafe_allow_html=True)

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

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    "<div class='footer'>Made with ❤️ using <a href='https://streamlit.io' target='_blank'>Streamlit</a> | © 2025 Mind Lens</div>",
    unsafe_allow_html=True
)
