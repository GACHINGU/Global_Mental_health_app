import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from deep_translator import GoogleTranslator
import pandas as pd
import time
import datetime
from io import BytesIO
import speech_recognition as sr

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Mind Lens — Futuristic", layout="centered", initial_sidebar_state="collapsed")

# ---------------------------
# Load model and tokenizer
# ---------------------------
@st.cache_resource
def load_model_and_tokenizer():
    repo_id = "Legend092/roberta-mentalhealth"
    model = RobertaForSequenceClassification.from_pretrained(repo_id)
    tokenizer = RobertaTokenizer.from_pretrained(repo_id)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# ---------------------------
# Label Mapping
# ---------------------------
label_mapping = {
    0: "anxiety",
    1: "bipolar",
    2: "depression",
    3: "normal",
    4: "personality disorder",
    5: "stress",
    6: "suicidal"
}

# ---------------------------
# Helpful Resources
# ---------------------------
resources = {
    "anxiety": [
        "Try slow breathing: inhale 4s, hold 4s, exhale 6s.",
        "Visit: https://www.anxietycentre.com",
        "Talk to a trusted friend or counselor."
    ],
    "depression": [
        "You’re not alone — reaching out helps more than you think.",
        "Call your local helpline or message a friend.",
        "Resource: https://findahelpline.com"
    ],
    "stress": [
        "Take a short walk or stretch for 5 minutes.",
        "Practice deep breathing or listen to calm music.",
        "Resource: https://www.stress.org"
    ],
    "bipolar": [
        "Track your mood daily to notice patterns.",
        "Keep routines consistent — especially sleep.",
        "Learn more: https://www.nami.org"
    ],
    "personality disorder": [
        "Connecting with a therapist can help you understand yourself.",
        "Try journaling to track emotions and triggers.",
        "Info: https://www.mind.org.uk"
    ],
    "suicidal": [
        "If you feel unsafe, please reach out now.",
        "Find help worldwide: https://findahelpline.com",
        "In Kenya: Befrienders Kenya – 0722 178177",
        "In the US: 988 Suicide & Crisis Lifeline"
    ],
    "normal": [
        "You seem balanced right now — keep practicing healthy habits.",
        "Maintain connections and regular breaks for mental wellness."
    ]
}

# ---------------------------
# Session State for mood tracking
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------
# Theme Switcher
# ---------------------------
theme_choice = st.sidebar.selectbox("Choose Theme", ["Futuristic Dark", "Calm Light"])

if theme_choice == "Futuristic Dark":
    background_css = """
    background: linear-gradient(180deg, #030416 0%, #041229 35%, #001428 70%, #000814 100%);
    color: #e6f7ff;
    """
else:
    background_css = """
    background: linear-gradient(180deg, #f0f4f8 0%, #d9e2ec 50%, #bcccdc 100%);
    color: #0b3d91;
    """

st.markdown(f"<style>.stApp {{{background_css}}}</style>", unsafe_allow_html=True)

# ---------------------------
# Title & Subtitle
# ---------------------------
st.markdown("<h1 style='text-align:center;'>MIND LENS</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Step in, let your words speak. Explore emotions, find balance, and connect with care.</p>", unsafe_allow_html=True)

# ---------------------------
# Audio Input Option
# ---------------------------
use_audio = st.checkbox("Use Audio Input")
if use_audio:
    st.write("Click to record your voice (speech-to-text)")
    audio_file = st.file_uploader("Upload or Record Audio", type=["wav","mp3"])
    if audio_file:
        recognizer = sr.Recognizer()
        with sr.AudioFile(BytesIO(audio_file.read())) as source:
            audio_data = recognizer.record(source)
            try:
                user_text = recognizer.recognize_google(audio_data)
                st.text_area("Detected Text:", user_text, height=170)
            except:
                st.warning("Audio could not be recognized. Please type instead.")
                user_text = st.text_area("Type your text here:", height=170)
    else:
        user_text = st.text_area("Type your text here:", height=170)
else:
    user_text = st.text_area("Type your text here:", height=170)

# ---------------------------
# Quick questionnaire for personalized tips
# ---------------------------
st.markdown("### Quick Check-in")
sleep_hours = st.slider("How many hours did you sleep last night?", 0, 12, 7)
stress_level = st.slider("Current stress level (1-10)", 1, 10, 5)
social_support = st.slider("Feeling socially supported? (1-10)", 1, 10, 5)

# ---------------------------
# Analyze Button
# ---------------------------
if st.button("Analyze"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        # ---------------------------
        # Translation
        # ---------------------------
        try:
            english_text = GoogleTranslator(source='auto', target='en').translate(user_text)
            st.info("Text translated to English.")
        except:
            english_text = user_text
            st.warning("Translation unavailable. Using original text.")

        # ---------------------------
        # Model Prediction
        # ---------------------------
        inputs = tokenizer(english_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            pred_class = torch.argmax(outputs.logits, dim=1).item()
            probs = torch.softmax(outputs.logits, dim=1)
            confidence = torch.max(probs).item()

        label = label_mapping.get(pred_class, "Unknown")

        # ---------------------------
        # Text Insights
        # ---------------------------
        keywords = [word for word in english_text.split() if word.lower() in label]
        insights = ", ".join(keywords) if keywords else "No specific keywords detected."

        # ---------------------------
        # Save to history
        # ---------------------------
        st.session_state.history.append({
            "datetime": datetime.datetime.now(),
            "text": user_text,
            "prediction": label,
            "confidence": confidence,
            "sleep_hours": sleep_hours,
            "stress_level": stress_level,
            "social_support": social_support
        })

        # ---------------------------
        # Show Results
        # ---------------------------
        st.success(f"Predicted Category: {label.upper()} (Confidence: {confidence*100:.1f}%)")
        st.info(f"Text Insights: {insights}")

        # Personalized Tips
        st.subheader("Helpful Suggestions")
        tips = resources.get(label, [])
        # Adjust tips based on questionnaire
        if sleep_hours < 6:
            tips.append("Try to get at least 6-7 hours of sleep tonight.")
        if stress_level > 7:
            tips.append("Take a 10-minute mindfulness break to reduce stress.")
        if social_support < 5:
            tips.append("Reach out to a friend or family member for support.")

        for tip in tips:
            st.markdown(f"- {tip}")

        # ---------------------------
        # Helpline Buttons
        # ---------------------------
        st.subheader("Immediate Help")
        if label == "suicidal":
            st.markdown("<a href='tel:+0722178177'><button>Call Befrienders Kenya</button></a>", unsafe_allow_html=True)
            st.markdown("<a href='tel:988'><button>Call US Lifeline</button></a>", unsafe_allow_html=True)

# ---------------------------
# Mood History Visualization
# ---------------------------
if st.session_state.history:
    st.subheader("Mood History")
    df_history = pd.DataFrame(st.session_state.history)
    st.line_chart(df_history["confidence"])
    st.dataframe(df_history[["datetime","text","prediction","confidence","sleep_hours","stress_level","social_support"]])

# ---------------------------
# Footer
# ---------------------------
st.markdown("<hr><p style='text-align:center;'>Made with care • Mind Lens • 2025</p>", unsafe_allow_html=True)