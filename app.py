import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from deep_translator import GoogleTranslator
import pandas as pd
import datetime
import sqlite3
import time

# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(
    page_title="Mind Lens — Futuristic",
    layout="centered",
    initial_sidebar_state="collapsed"
)

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
# Label mapping & resources
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
# Database setup
# ---------------------------
conn = sqlite3.connect("mind_lens.db", check_same_thread=False)
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS user_moods (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        timestamp TEXT,
        text TEXT,
        prediction TEXT,
        confidence REAL,
        sleep_hours INTEGER,
        stress_level INTEGER,
        social_support INTEGER
    )
''')
conn.commit()

# ---------------------------
# User login / identification
# ---------------------------
user_id = st.sidebar.text_input("Enter your username", value="guest")

# ---------------------------
# Load user history
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if user_id:
    query = "SELECT * FROM user_moods WHERE user_id = ? ORDER BY timestamp"
    user_df = pd.read_sql_query(query, conn, params=(user_id,))
    if not user_df.empty:
        st.session_state.history = user_df.to_dict('records')

# ---------------------------
# Futuristic Dark-Blue Theme
# ---------------------------
st.markdown("""
<style>
.stApp {background: linear-gradient(180deg, #030416 0%, #041229 35%, #001428 70%, #000814 100%); color: #e6f7ff; font-family: "Inter", "Segoe UI", Roboto, sans-serif; min-height: 100vh;}
.glow-anim {position: fixed; inset: 0; pointer-events: none; z-index: 0; background: radial-gradient(800px 300px at 10% 20%, rgba(0,160,255,0.06), transparent 10%), radial-gradient(600px 240px at 90% 80%, rgba(0,120,255,0.05), transparent 12%); animation: slowPulse 8s ease-in-out infinite; mix-blend-mode: screen;}
@keyframes slowPulse {0% { opacity: 0.85; transform: scale(1); } 50% { opacity: 1; transform: scale(1.02); } 100% { opacity: 0.86; transform: scale(1); }}
.card {position: relative; z-index: 1; width: 860px; max-width: 96%; margin: 28px auto; background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius: 14px; padding: 24px; box-shadow: 0 20px 60px rgba(0,4,12,0.75), inset 0 1px 0 rgba(255,255,255,0.02); border: 1px solid rgba(0,200,255,0.06); backdrop-filter: blur(6px); transform: translateY(6px); animation: floatUp 0.7s ease forwards;}
@keyframes floatUp { from { opacity:0; transform:translateY(20px);} to {opacity:1; transform:translateY(0);} }
.title {text-align:center; font-size:36px; font-weight:800; letter-spacing:1.6px; margin-bottom:6px; background: linear-gradient(90deg, #bff9ff, #00d4ff 36%, #00a0ff 76%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; filter: drop-shadow(0 8px 30px rgba(0,160,255,0.12));}
.subtitle {text-align:center; color:#9db9c8; margin-bottom:18px; font-size:14.5px;}
textarea {background-color: #06131b !important; color: #eafcff !important; border-radius: 12px !important; border: 1px solid rgba(0,160,255,0.18) !important; padding: 14px !important; font-size: 15.5px !important; box-shadow: 0 8px 24px rgba(0,160,255,0.02); transition: box-shadow 0.18s ease, transform 0.12s ease;}
textarea:focus {box-shadow: 0 0 28px rgba(0,170,255,0.16) !important; transform: translateY(-2px); outline: none !important;}
div.stButton > button {background: linear-gradient(90deg, rgba(0,210,255,0.12), rgba(0,160,255,0.08)); color: #eaffff; border: 1px solid rgba(0,160,255,0.6); border-radius: 12px; padding: 12px 18px; font-weight: 800; font-size: 15.5px; letter-spacing: 0.6px; transition: transform 0.16s ease, box-shadow 0.16s ease;}
div.stButton > button:hover {transform: translateY(-5px) scale(1.01); box-shadow: 0 20px 60px rgba(0,160,255,0.12), 0 0 40px rgba(0,160,255,0.06) inset;}
h3,h4,p,li {color: #d6f2fb; font-weight:700;}
hr {border:0; height:1px; background: linear-gradient(90deg, rgba(0,160,255,0.12), rgba(255,255,255,0.02)); margin:18px 0;}
.footer {text-align:center; color:#7f98a6; font-size:13px; margin-top:14px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='glow-anim'></div>", unsafe_allow_html=True)

# ---------------------------
# Sidebar Navigation
# ---------------------------
page = st.sidebar.selectbox(
    "Navigate",
    ["Home", "Mood History", "Category Trends", "Global Insights"]
)

# ---------------------------
# Page 1: Home / Analysis
# ---------------------------
if page == "Home":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='title'>MIND LENS</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Step in, let your words speak. Explore emotions, find balance, and connect with care.</div>", unsafe_allow_html=True)

    user_text = st.text_area("Type your text here:", height=170)
    st.markdown("### Quick Check-in")
    sleep_hours = st.slider("Hours slept last night?", 0, 12, 7)
    stress_level = st.slider("Current stress level (1-10)", 1, 10, 5)
    social_support = st.slider("Feeling socially supported? (1-10)", 1, 10, 5)

    if st.button("Analyze"):
        if not user_text.strip():
            st.warning("Please enter some text.")
        else:
            try:
                english_text = GoogleTranslator(source='auto', target='en').translate(user_text)
                st.info("Text translated to English.")
            except:
                english_text = user_text
                st.warning("Translation unavailable. Using original text.")

            inputs = tokenizer(english_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
                pred_class = torch.argmax(outputs.logits, dim=1).item()
                probs = torch.softmax(outputs.logits, dim=1)
                confidence = torch.max(probs).item()

            label = label_mapping.get(pred_class, "Unknown")
            insights = ", ".join([word for word in english_text.split() if word.lower() in label]) or "No specific keywords detected."

            # Save to session state
            st.session_state.history.append({
                "datetime": datetime.datetime.now(),
                "text": user_text,
                "prediction": label,
                "confidence": confidence,
                "sleep_hours": sleep_hours,
                "stress_level": stress_level,
                "social_support": social_support
            })

            # Save to database
            c.execute('''
                INSERT INTO user_moods (user_id, timestamp, text, prediction, confidence, sleep_hours, stress_level, social_support)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, datetime.datetime.now(), user_text, label, confidence, sleep_hours, stress_level, social_support))
            conn.commit()

            st.success(f"Predicted Category: {label.upper()} (Confidence: {confidence*100:.1f}%)")
            st.info(f"Text Insights: {insights}")

            st.subheader("Helpful Suggestions")
            tips = resources.get(label, [])
            if sleep_hours < 6: tips.append("Try to get at least 6-7 hours of sleep tonight.")
            if stress_level > 7: tips.append("Take a 10-minute mindfulness break to reduce stress.")
            if social_support < 5: tips.append("Reach out to a friend or family member for support.")
            for tip in tips: st.markdown(f"- {tip}")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Page 2: Mood History
# ---------------------------
elif page == "Mood History":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='title'>Mood History</div>", unsafe_allow_html=True)
    if st.session_state.history:
        df_history = pd.DataFrame(st.session_state.history)
        st.line_chart(df_history["confidence"])
        st.dataframe(df_history[["datetime","text","prediction","confidence","sleep_hours","stress_level","social_support"]])
        csv = df_history.to_csv(index=False).encode('utf-8')
        st.download_button("Download Mood & Analysis History", data=csv, file_name='mind_lens_history.csv', mime='text/csv')
    else:
        st.info("No mood history yet.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Page 3: Category Trends
# ---------------------------
elif page == "Category Trends":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='title'>Category Trends</div>", unsafe_allow_html=True)
    df_history = pd.DataFrame(st.session_state.history)
    if not df_history.empty:
        st.bar_chart(df_history['prediction'].value_counts())
    else:
        st.info("No data yet for trends.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Page 4: Global Insights
# ---------------------------
elif page == "Global Insights":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='title'>Global Insights</div>", unsafe_allow_html=True)
    global_df = pd.read_sql_query("SELECT * FROM user_moods", conn)
    if not global_df.empty:
        st.subheader("Most Diagnosed Categories")
        st.bar_chart(global_df['prediction'].value_counts())

        st.subheader("Average Stress by Category")
        avg_stress = global_df.groupby('prediction')['stress_level'].mean()
        st.bar_chart(avg_stress)

        st.subheader("Category Submission Trends Over Time")
        global_df['timestamp'] = pd.to_datetime(global_df['timestamp'])
        global_df['date'] = global_df['timestamp'].dt.date
        trend_by_category = global_df.groupby(['date', 'prediction']).size().unstack(fill_value=0)
        st.line_chart(trend_by_category)
    else:
        st.info("No global data available yet.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown("<div class='footer'>Made with care • Mind Lens • 2025</div>", unsafe_allow_html=True)