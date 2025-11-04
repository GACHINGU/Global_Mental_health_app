import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from deep_translator import GoogleTranslator
import time

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Mind Lens — Futuristic", layout="centered", initial_sidebar_state="collapsed")

# ------------------------------------------------
# Load model and tokenizer directly from Hugging Face
# (UNCHANGED logic)
# ------------------------------------------------
@st.cache_resource
def load_model_and_tokenizer():
    repo_id = "Legend092/roberta-mentalhealth"
    model = RobertaForSequenceClassification.from_pretrained(repo_id)
    tokenizer = RobertaTokenizer.from_pretrained(repo_id)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# ------------------------------------------------
# Label Mapping (UNCHANGED)
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
# Helpful Resources (UNCHANGED)
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
# FUTURISTIC DARK->BLUE ANIMATED GLOW THEME
# ------------------------------------------------
st.markdown(
    """
    <style>
    /* Base page */
    .stApp {
        background: linear-gradient(180deg, #030416 0%, #041229 35%, #001428 70%, #000814 100%);
        color: #e6f7ff;
        font-family: "Inter", "Segoe UI", Roboto, sans-serif;
        min-height: 100vh;
    }

    /* animated glow overlay */
    .glow-anim {
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        background:
            radial-gradient(800px 300px at 10% 20%, rgba(0,160,255,0.06), transparent 10%),
            radial-gradient(600px 240px at 90% 80%, rgba(0,120,255,0.05), transparent 12%);
        animation: slowPulse 8s ease-in-out infinite;
        mix-blend-mode: screen;
    }
    @keyframes slowPulse {
        0% { opacity: 0.85; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.02); }
        100% { opacity: 0.86; transform: scale(1); }
    }

    /* container card */
    .card {
        position: relative;
        z-index: 1;
        width: 860px;
        max-width: 96%;
        margin: 28px auto;
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border-radius: 14px;
        padding: 24px;
        box-shadow:
            0 20px 60px rgba(0,4,12,0.75),
            inset 0 1px 0 rgba(255,255,255,0.02);
        border: 1px solid rgba(0,200,255,0.06);
        backdrop-filter: blur(6px);
    }

    /* holographic title */
    .title {
        display:block;
        text-align:center;
        font-size:36px;
        font-weight:800;
        letter-spacing:1.6px;
        margin-bottom:6px;
        background: linear-gradient(90deg, #bff9ff, #00d4ff 36%, #00a0ff 76%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 8px 30px rgba(0,160,255,0.12));
    }

    .subtitle {
        text-align:center;
        color:#9db9c8;
        margin-bottom:18px;
        font-size:14.5px;
    }

    /* neon text area */
    textarea {
        background-color: #06131b !important;
        color: #eafcff !important;
        border-radius: 12px !important;
        border: 1px solid rgba(0,160,255,0.18) !important;
        padding: 14px !important;
        font-size: 15.5px !important;
        box-shadow: 0 8px 24px rgba(0,160,255,0.02);
        transition: box-shadow 0.18s ease, transform 0.12s ease;
    }
    textarea:focus {
        box-shadow: 0 0 28px rgba(0,170,255,0.16) !important;
        transform: translateY(-2px);
        outline: none !important;
    }

    /* neon button */
    div.stButton > button {
        background: linear-gradient(90deg, rgba(0,210,255,0.12), rgba(0,160,255,0.08));
        color: #eaffff;
        border: 1px solid rgba(0,160,255,0.6);
        border-radius: 12px;
        padding: 12px 18px;
        font-weight: 800;
        font-size: 15.5px;
        letter-spacing: 0.6px;
        transition: transform 0.16s ease, box-shadow 0.16s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-5px) scale(1.01);
        box-shadow: 0 20px 60px rgba(0,160,255,0.12), 0 0 40px rgba(0,160,255,0.06) inset;
    }

    /* result headings */
    h3, h4 { color: #cfeff7; font-weight:700; }
    p, li { color: #d6f2fb; }

    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, rgba(0,160,255,0.12), rgba(255,255,255,0.02));
        margin: 18px 0;
    }

    .footer {
        text-align:center;
        color:#7f98a6;
        font-size:13px;
        margin-top:14px;
    }

    /* scanner loader elements (in-card) */
    .scanner {
        width:100%;
        height:56px;
        display:flex;
        align-items:center;
        justify-content:center;
        gap:14px;
    }
    .orb {
        width:18px;
        height:18px;
        border-radius:50%;
        background: radial-gradient(circle at 30% 30%, #bff9ff, #00d4ff 50%, #008cff 100%);
        box-shadow: 0 0 22px rgba(0,212,255,0.45), 0 0 60px rgba(0,160,255,0.08);
        animation: orbPulse 1.4s linear infinite;
    }
    .bar {
        width:62%;
        height:8px;
        border-radius:999px;
        background: linear-gradient(90deg, rgba(0,160,255,0.14), rgba(0,212,255,0.22));
        position:relative;
        overflow:hidden;
    }
    .bar::before {
        content: "";
        position:absolute;
        left:-28%;
        width:30%;
        height:100%;
        background: linear-gradient(90deg, rgba(255,255,255,0.06), rgba(255,255,255,0.24), rgba(255,255,255,0.06));
        transform: skewX(-25deg);
        animation: sweep 1.5s ease-in-out infinite;
    }
    @keyframes sweep { 0% { left:-28%; } 50% { left:84%; } 100% { left:-28%; } }
    @keyframes orbPulse { 0% { transform:scale(0.9); opacity:0.85 } 50% { transform:scale(1.15); opacity:1 } 100% { transform:scale(0.9); opacity:0.85 } }

    /* floating card animation */
    .card { transform: translateY(6px); animation: floatUp 0.7s ease forwards; }
    @keyframes floatUp { from { opacity:0; transform:translateY(20px);} to {opacity:1; transform:translateY(0);} }

    /* responsive tweaks */
    @media (max-width: 880px) {
        .card { width:94% !important; padding:18px; }
        .title { font-size:28px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# animated glow overlay element
st.markdown("<div class='glow-anim'></div>", unsafe_allow_html=True)

# ---------------------------
# Layout card start
# ---------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center'><span class='title'>MIND LENS</span></div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Step in, let your words speak. Explore emotions, find balance, and connect with care wherever you are.</div>", unsafe_allow_html=True)

# Input area (functionality preserved)
user_text = st.text_area("💬 Type or paste your text here:", height=170)

# Analyze button
analyze_clicked = st.button("🔎 ANALYZE (SCAN)")

if analyze_clicked:
    if not user_text.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        # show scanning loader HTML while processing
        loader = st.empty()
        loader.markdown(
            """
            <div class="scanner">
                <div class="orb"></div>
                <div class="bar"></div>
            </div>
            <div style="text-align:center;color:#9fbfcf;font-size:13px;margin-top:-8px;">Scanning input & running analysis...</div>
            """,
            unsafe_allow_html=True,
        )

        # brief UX pause so the scan feels deliberate
        time.sleep(0.6)

        # TRANSLATION (unchanged)
        try:
            english_text = GoogleTranslator(source='auto', target='en').translate(user_text)
            st.info("🌍 Text has been translated to English (if needed).")
            st.markdown(f"**Translated text:** {english_text}")
        except Exception:
            english_text = user_text
            st.warning("⚠️ Translation service unavailable — using original text.")

        # MODEL PREDICTION (unchanged)
        inputs = tokenizer(english_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            pred_class = torch.argmax(outputs.logits, dim=1).item()

        label = label_mapping.get(pred_class, "Unknown")

        # small UX pause
        time.sleep(0.45)

        # remove loader and show results
        loader.empty()

        st.success(f"🩺 Predicted Mental Health Category: **{label.upper()}**")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("💡 Helpful Suggestions & Resources:")
        for tip in resources.get(label, []):
            st.markdown(f"- {tip}")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.caption("⚠️ This tool is for informational support only and does not replace professional mental health advice.")
        st.caption("⚠️ Translations may not be perfect; always seek local professional help when needed.")

# close card
st.markdown("</div>", unsafe_allow_html=True)

# footer
st.markdown("<div class='footer'>Made with ❤️ • Mind Lens • 2025</div>", unsafe_allow_html=True)
