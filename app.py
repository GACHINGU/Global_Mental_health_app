import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from deep_translator import GoogleTranslator

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
        "Visit: [anxietycentre.com](https://www.anxietycentre.com)",
        "Talk to a trusted friend or counselor."
    ],
    "depression": [
        "You’re not alone — reaching out helps more than you think.",
        "Call your local helpline or message a friend.",
        "Resource: [findahelpline.com](https://findahelpline.com)"
    ],
    "stress": [
        "Take a short walk or stretch for 5 minutes.",
        "Listen to calm music or practice deep breathing.",
        "Resource: [stress.org](https://www.stress.org)"
    ],
    "bipolar": [
        "Track your mood daily to notice patterns.",
        "Keep routines consistent — especially sleep.",
        "Learn more: [nami.org](https://www.nami.org)"
    ],
    "personality disorder": [
        "Therapy can help you understand yourself better.",
        "Try journaling to track emotions and triggers.",
        "Info: [mind.org.uk](https://www.mind.org.uk)"
    ],
    "suicidal": [
        "If you feel unsafe, **please reach out now**.",
        "Find help: [findahelpline.com](https://findahelpline.com)",
        "🇰🇪 Kenya: Befrienders Kenya – 0722 178177",
        "🇺🇸 US: 988 Suicide & Crisis Lifeline"
    ],
    "normal": [
        "You seem balanced — keep practicing healthy habits!",
        "Maintain connections and take regular breaks."
    ]
}

# ------------------------------------------------
# Page Setup & CSS
# ------------------------------------------------
st.set_page_config(page_title="Mind Lens 🔍", layout="centered")

st.markdown("""
    <style>
    /* Background with glowing brain theme */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),
                    url("https://images.unsplash.com/photo-1509099836639-18ba1795216d?auto=format&fit=crop&w=1950&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        color: #000;
    }

    /* Fonts and text styling */
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    h1 {
        color: #ffffff;
        text-align: center;
        font-size: 2.6em;
        margin-top: 10px;
    }

    h2, h3, p, label {
        color: #f0f0f0 !important;
    }

    /* Input box styling */
    textarea {
        background-color: rgba(255,255,255,0.15) !important;
        color: #fff !important;
        border-radius: 12px !important;
        border: 1px solid #ffffff44 !important;
    }

    /* Buttons */
    div.stButton > button:first-child {
        background-color: #00bfa6;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.7em 1.3em;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    div.stButton > button:first-child:hover {
        background-color: #00a38d;
        transform: scale(1.05);
    }

    /* Card-style resource section */
    .resource-box {
        background-color: rgba(255,255,255,0.1);
        padding: 1em;
        border-radius: 12px;
        margin-top: 1em;
        backdrop-filter: blur(10px);
    }

    .footer {
        color: #dddddd;
        text-align: center;
        font-size: 0.85em;
        margin-top: 2em;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Main UI
# ------------------------------------------------
st.markdown("<h1>🧠 Mind Lens — Discover Your Emotional Landscape</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Let your words speak — explore emotions, find balance, and connect with care 💬</p>", unsafe_allow_html=True)

user_text = st.text_area("Type or paste your text here:", height=150, placeholder="Write your thoughts here...")

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

        # Model prediction
        inputs = tokenizer(english_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            pred_class = torch.argmax(outputs.logits, dim=1).item()

        label = label_mapping.get(pred_class, "Unknown")

        st.markdown(f"<h2 style='color:#00ffc6;'>💭 Predicted Mental Health Category: {label.upper()}</h2>", unsafe_allow_html=True)
        st.markdown("<div class='resource-box'>", unsafe_allow_html=True)
        st.subheader("💬 Helpful Suggestions & Resources:")
        for tip in resources.get(label, []):
            st.markdown(f"- {tip}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='footer'>⚠️ This tool is for informational support only and does not replace professional mental health advice.<br>Translations may not be perfect; always seek local professional help when needed.</div>", unsafe_allow_html=True)