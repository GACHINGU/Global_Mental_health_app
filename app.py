import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from deep_translator import GoogleTranslator
import time

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
# FUTURISTIC THEME + SCANNING LOADER (styling only)
# ------------------------------------------------
st.set_page_config(page_title="Mind Lens — Futuristic", layout="centered")

st.markdown(
    """
    <style>
    /* PAGE BACKGROUND */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #071426, #031014 40%, #00060a 100%);
        color: #e6f7ff;
        font-family: "Inter", "Segoe UI", Roboto, sans-serif;
    }

    /* Holographic title */
    .title {
        font-size: 36px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 6px;
        background: linear-gradient(90deg, #8ef0ff, #00d4ff 40%, #00a0ff 75%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 6px 18px rgba(0,160,255,0.18));
    }

    .subtitle {
        text-align: center;
        color: #9fbfcf;
        margin-bottom: 18px;
        font-size: 14.5px;
    }

    /* Center floating card */
    .card {
        width: 820px;
        max-width: 94%;
        margin: 18px auto;
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
        border-radius: 14px;
        padding: 22px;
        box-shadow:
            0 8px 30px rgba(0,0,0,0.6),
            inset 0 1px 0 rgba(255,255,255,0.02);
        border: 1px solid rgba(0,200,255,0.06);
        backdrop-filter: blur(6px);
    }

    /* Text area: dark with cyan glow */
    textarea {
        background-color: #071420 !important;
        color: #eafcff !important;
        border-radius: 12px !important;
        border: 1px solid rgba(0,160,255,0.18) !important;
        padding: 12px !important;
        font-size: 15.5px !important;
        box-shadow: 0 6px 18px rgba(0,160,255,0.03);
    }
    textarea:focus {
        box-shadow: 0 0 28px rgba(0,170,255,0.18) !important;
        outline: none !important;
    }

    /* Button: neon */
    div.stButton > button {
        background: linear-gradient(90deg, rgba(0, 210, 255, 0.14), rgba(0, 160, 255, 0.14));
        color: #eaffff;
        border: 1px solid rgba(0,160,255,0.6);
        border-radius: 12px;
        padding: 12px 18px;
        font-weight: 700;
        font-size: 15.5px;
        transition: transform 0.16s ease, box-shadow 0.16s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-4px) scale(1.01);
        box-shadow: 0 10px 30px rgba(0,160,255,0.18), 0 0 40px rgba(0,160,255,0.06) inset;
    }

    /* Results & resource headers */
    h3, h4 {
        color: #bfefff;
        font-weight: 700;
    }
    p, li {
        color: #cfeff7;
    }

    /* separators */
    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, rgba(0,160,255,0.12), rgba(255,255,255,0.02));
        margin: 18px 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 13px;
        color: #7f98a6;
        margin-top: 12px;
    }

    /* --- SCANNING LOADER --- */
    .scanner {
        width: 100%;
        height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .scanner .orb {
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: radial-gradient(circle at 30% 30%, #bff9ff, #00d4ff 50%, #008cff 100%);
        box-shadow: 0 0 20px rgba(0,212,255,0.45), 0 0 60px rgba(0,160,255,0.08);
        animation: pulse 1.4s linear infinite;
        margin-right: 12px;
    }
    .scanner .bar {
        width: 60%;
        height: 6px;
        border-radius: 999px;
        background: linear-gradient(90deg, rgba(0,160,255,0.2), rgba(0,212,255,0.3));
        position: relative;
        overflow: hidden;
    }
    .scanner .bar::before {
        content: "";
        position: absolute;
        left: -30%;
        width: 30%;
        height: 100%;
        background: linear-gradient(90deg, rgba(255,255,255,0.06), rgba(255,255,255,0.25), rgba(255,255,255,0.06));
        transform: skewX(-25deg);
        animation: sweep 1.6s ease-in-out infinite;
    }
    @keyframes sweep {
        0% { left: -30%; }
        50% { left: 80%; }
        100% { left: -30%; }
    }
    @keyframes pulse {
        0% { transform: scale(0.9); opacity: 0.85; }
        50% { transform: scale(1.16); opacity: 1; }
        100% { transform: scale(0.9); opacity: 0.85; }
    }

    /* subtle floating card animation on load */
    .card { transform: translateY(6px); animation: floatUp 0.7s ease forwards; }
    @keyframes floatUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }

    /* Responsive */
    @media (max-width: 800px) {
        .card { width: 94% !important; padding: 18px; }
        h1 { font-size: 28px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------
# PAGE LAYOUT
# ------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center'><h1 class='title'>MIND LENS</h1></div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Futuristic AI dashboard — type your text and let the scanner analyze emotional tone.</div>", unsafe_allow_html=True)

# Input area (kept functionality)
user_text = st.text_area("💬 Type or paste your text here:", height=170)

# BUTTON + SCANNER behaviour:
analyze_clicked = st.button("🔎 ANALYZE (SCAN)")

if analyze_clicked:
    if not user_text.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        # show scanning loader (custom HTML) while processing
        loader_placeholder = st.empty()
        loader_placeholder.markdown(
            """
            <div class="scanner">
                <div class="orb"></div>
                <div class="bar"></div>
            </div>
            <div style="text-align:center;color:#9fbfcf;font-size:13px;margin-top:-8px;">Scanning input & running analysis...</div>
            """,
            unsafe_allow_html=True,
        )

        # small artificial wait for UX (keeps loader visible for a moment)
        time.sleep(0.6)

        # --- TRANSLATE (unchanged) ---
        try:
            english_text = GoogleTranslator(source='auto', target='en').translate(user_text)
            st.info("🌍 Text has been translated to English (if needed).")
            st.markdown(f"**Translated text:** {english_text}")
        except Exception:
            english_text = user_text
            st.warning("⚠️ Translation service unavailable — using original text.")

        # --- MODEL PREDICTION (unchanged) ---
        inputs = tokenizer(english_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            pred_class = torch.argmax(outputs.logits, dim=1).item()

        label = label_mapping.get(pred_class, "Unknown")

        # small UX pause so scanning feels deliberate
        time.sleep(0.45)

        # remove loader and show results (keep same content)
        loader_placeholder.empty()

        st.success(f"🩺 Predicted Mental Health Category: **{label.upper()}**")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("💡 Helpful Suggestions & Resources:")
        for tip in resources.get(label, []):
            st.markdown(f"- {tip}")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.caption("⚠️ This tool is for informational support only and does not replace professional mental health advice.")
        st.caption("⚠️ Translations may not be perfect; always seek local professional help when needed.")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>Made with ❤️ • Mind Lens • 2025</div>", unsafe_allow_html=True)
