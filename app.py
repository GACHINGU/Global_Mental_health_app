import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sqlite3
import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from deep_translator import GoogleTranslator
import time
import hashlib
import binascii
import pandas as pd
from datetime import datetime, timedelta
import json

# ---------------------------
# Configuration
# ---------------------------
DB_PATH = "mind_lens.db"
SITE_NAME_DEFAULT = "Mind Lens"
DEFAULT_REFRESH_SECONDS = 10

# ---------------------------
# Utilities: DB and security
# ---------------------------

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    # users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
    ''')
    # results table (also used for visit logging)
    c.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            text TEXT,
            translated_text TEXT,
            label TEXT,
            timestamp TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    # settings table
    c.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    conn.commit()
    conn.close()


# password hashing using PBKDF2
def hash_password(password: str, salt: bytes = None) -> str:
    if salt is None:
        salt = os.urandom(16)
    pwd = password.encode('utf-8')
    dk = hashlib.pbkdf2_hmac('sha256', pwd, salt, 100000)
    return binascii.hexlify(salt).decode() + ':' + binascii.hexlify(dk).decode()


def verify_password(stored_hash: str, provided_password: str) -> bool:
    try:
        salt_hex, dk_hex = stored_hash.split(':')
        salt = binascii.unhexlify(salt_hex)
        pwd = provided_password.encode('utf-8')
        new_dk = hashlib.pbkdf2_hmac('sha256', pwd, salt, 100000)
        return binascii.hexlify(new_dk).decode() == dk_hex
    except Exception:
        return False


# settings helpers
def get_setting(key: str, default=None):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT value FROM settings WHERE key=?', (key,))
    row = c.fetchone()
    conn.close()
    if row:
        return row['value']
    return default


def set_setting(key: str, value: str):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('REPLACE INTO settings(key, value) VALUES(?, ?)', (key, value))
    conn.commit()
    conn.close()


# ---------------------------
# Initialize DB and default admin
# ---------------------------
init_db()
# create default admin if not exists
conn = get_db_connection()
c = conn.cursor()
c.execute('SELECT * FROM users WHERE username=?', ('admin',))
if not c.fetchone():
    pwd_hash = hash_password('admin123')
    c.execute('INSERT INTO users(username, password_hash, is_admin, created_at) VALUES(?, ?, ?, ?)',
              ('admin', pwd_hash, 1, datetime.utcnow().isoformat()))
    conn.commit()
conn.close()

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title=SITE_NAME_DEFAULT + " — Futuristic", layout="centered", initial_sidebar_state="expanded")

# ---------------------------
# Load model and tokenizer (cached)
# ---------------------------
@st.cache_resource
def load_model_and_tokenizer():
    repo_id = "Legend092/roberta-mentalhealth"
    try:
        model = RobertaForSequenceClassification.from_pretrained(repo_id)
        tokenizer = RobertaTokenizer.from_pretrained(repo_id)
    except Exception:
        st.warning("Model could not be loaded from the remote repository. Ensure internet access or that the repo ID is correct.")
        model, tokenizer = None, None
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# label mapping
label_mapping = {
    0: "anxiety",
    1: "bipolar",
    2: "depression",
    3: "normal",
    4: "personality disorder",
    5: "stress",
    6: "suicidal"
}

# resources
resources = {
    "anxiety": [
        "Try slow breathing: inhale 4s, hold 4s, exhale 6s.",
        "Visit: https://www.anxietycentre.com",
        "Talk to a trusted friend or counselor."
    ],
    "depression": [
        "You are not alone — reaching out helps more than you think.",
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
        "You seem balanced right now — keep practicing healthy habits!",
        "Maintain connections and regular breaks for mental wellness."
    ]
}

# ---------------------------
# Simple CSS (no emojis anywhere)
# ---------------------------
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #030416 0%, #041229 35%, #001428 70%, #000814 100%); color: #e6f7ff; font-family: "Inter", "Segoe UI", Roboto, sans-serif; min-height: 100vh; }
    .card { position: relative; z-index: 1; width: 860px; max-width: 96%; margin: 18px auto; background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.005)); border-radius: 12px; padding: 18px; box-shadow: 0 10px 40px rgba(0,4,12,0.6); border: 1px solid rgba(0,200,255,0.04); }
    .title { display:block; text-align:center; font-size:28px; font-weight:800; letter-spacing:1.2px; margin-bottom:6px; color:#bff9ff }
    .subtitle { text-align:center; color:#9db9c8; margin-bottom:12px; font-size:13px }
    textarea { background-color: #06131b !important; color: #eafcff !important; border-radius: 10px !important; border: 1px solid rgba(0,160,255,0.12) !important; padding: 12px !important; font-size: 14px !important; }
    div.stButton > button { background: linear-gradient(90deg, rgba(0,210,255,0.08), rgba(0,160,255,0.05)); color: #eaffff; border: 1px solid rgba(0,160,255,0.35); border-radius: 10px; padding: 10px 14px; font-weight:700 }
    .footer { text-align:center; color:#7f98a6; font-size:12px; margin-top:12px }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Session helpers
# ---------------------------
if 'user' not in st.session_state:
    st.session_state['user'] = None
if 'is_admin' not in st.session_state:
    st.session_state['is_admin'] = False


def login_user(username: str):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT id, is_admin FROM users WHERE username=?', (username,))
    row = c.fetchone()
    conn.close()
    if row:
        st.session_state['user'] = {'id': row['id'], 'username': username}
        st.session_state['is_admin'] = bool(row['is_admin'])
        return True
    return False


def logout_user():
    st.session_state['user'] = None
    st.session_state['is_admin'] = False


# ---------------------------
# Navigation
# ---------------------------
st.sidebar.title(get_setting('site_name', SITE_NAME_DEFAULT) if get_setting('site_name') else SITE_NAME_DEFAULT)
menu = st.sidebar.selectbox("Navigation", ["Home", "Analyze", "Sign Up", "Login", "Dashboard", "Settings", "About", "Logout"])

# ---------------------------
# Home
# ---------------------------
if menu == "Home":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='title'>" + (get_setting('site_name', SITE_NAME_DEFAULT) or SITE_NAME_DEFAULT) + "</div>")
    st.markdown("<div class='subtitle'>Step in, let your words speak. Explore emotions, find balance, and connect with care wherever you are.</div>", unsafe_allow_html=True)
    st.write("This application analyzes short text and suggests likely mental health categories as an informational aid. It is not a replacement for professional help.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Sign Up
# ---------------------------
elif menu == "Sign Up":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Create an account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    password2 = st.text_input("Confirm password", type="password")
    if st.button("Create account"):
        if not username or not password:
            st.warning("Please provide both username and password.")
        elif password != password2:
            st.warning("Passwords do not match.")
        else:
            conn = get_db_connection()
            c = conn.cursor()
            try:
                pwd_hash = hash_password(password)
                c.execute('INSERT INTO users(username, password_hash, is_admin, created_at) VALUES(?, ?, ?, ?)',
                          (username, pwd_hash, 0, datetime.utcnow().isoformat()))
                conn.commit()
                st.success("Account created. You may now log in.")
            except sqlite3.IntegrityError:
                st.error("Username already exists. Choose a different username.")
            finally:
                conn.close()
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Login
# ---------------------------
elif menu == "Login":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if not username or not password:
            st.warning("Please enter username and password.")
        else:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute('SELECT password_hash FROM users WHERE username=?', (username,))
            row = c.fetchone()
            conn.close()
            if row and verify_password(row['password_hash'], password):
                login_user(username)
                st.success("Login successful.")
            else:
                st.error("Invalid username or password.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Logout
# ---------------------------
elif menu == "Logout":
    if st.session_state['user']:
        if st.button("Confirm logout"):
            logout_user()
            st.success("You have been logged out.")
    else:
        st.info("No user is currently logged in.")

# ---------------------------
# Analyze (user flow)
# ---------------------------
elif menu == "Analyze":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='title'>Analyze text</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Paste a text sample and run the analysis. Results can be downloaded.</div>", unsafe_allow_html=True)

    user_text = st.text_area("Type or paste your text here:", height=170)
    analyze_clicked = st.button("Analyze")

    if analyze_clicked:
        if not user_text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Running analysis..."):
                # translation
                try:
                    english_text = GoogleTranslator(source='auto', target='en').translate(user_text)
                    st.info("Text has been translated to English (if needed).")
                except Exception:
                    english_text = user_text
                    st.warning("Translation service unavailable — using original text.")

                # model prediction
                if model is None or tokenizer is None:
                    st.error("Model is not available. Prediction cannot be performed.")
                else:
                    try:
                        inputs = tokenizer(english_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
                        with torch.no_grad():
                            outputs = model(**inputs)
                            pred_class = torch.argmax(outputs.logits, dim=1).item()
                        label = label_mapping.get(pred_class, "Unknown")
                    except Exception:
                        label = "Unknown"

                # save result to DB (if logged in, associate with user)
                conn = get_db_connection()
                c = conn.cursor()
                user_id = st.session_state['user']['id'] if st.session_state['user'] else None
                timestamp = datetime.utcnow().isoformat()
                c.execute('INSERT INTO results(user_id, text, translated_text, label, timestamp) VALUES(?, ?, ?, ?, ?)',
                          (user_id, user_text, english_text, label, timestamp))
                conn.commit()
                conn.close()

                # show results
                st.success(f"Predicted Mental Health Category: {label.upper()}")
                st.markdown("---")
                st.subheader("Helpful Suggestions & Resources:")
                for tip in resources.get(label, []):
                    st.write(f"- {tip}")
                st.markdown("---")
                st.caption("This tool is for informational support only and does not replace professional mental health advice.")
                st.caption("Translations may not be perfect; always seek local professional help when needed.")

                # prepare download data
                result_row = {
                    'user': st.session_state['user']['username'] if st.session_state['user'] else 'anonymous',
                    'text': user_text,
                    'translated_text': english_text,
                    'label': label,
                    'timestamp': timestamp
                }
                df = pd.DataFrame([result_row])
                csv = df.to_csv(index=False).encode('utf-8')
                json_str = df.to_json(orient='records')

                st.download_button("Download result as CSV", csv, file_name='mind_lens_result.csv', mime='text/csv')
                st.download_button("Download result as JSON", json_str, file_name='mind_lens_result.json', mime='application/json')

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Dashboard (admin)
# ---------------------------
elif menu == "Dashboard":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Admin Dashboard")
    if not st.session_state['is_admin']:
        st.error("Access denied. Admins only.")
    else:
        conn = get_db_connection()
        c = conn.cursor()
        # basic stats
        c.execute('SELECT COUNT(*) as cnt FROM users')
        total_users = c.fetchone()['cnt']
        c.execute('SELECT COUNT(*) as cnt FROM results')
        total_results = c.fetchone()['cnt']

        st.metric("Total users", total_users)
        st.metric("Total analyses", total_results)

        # frequent visitors (by number of results)
        c.execute('''
            SELECT u.username, COUNT(r.id) as visits
            FROM users u
            LEFT JOIN results r ON u.id = r.user_id
            GROUP BY u.id
            ORDER BY visits DESC
            LIMIT 10
        ''')
        frequent = c.fetchall()
        if frequent:
            df_freq = pd.DataFrame(frequent)
            st.subheader("Frequent visitors")
            st.table(df_freq)
        else:
            st.info("No signed up users with visits yet.")

        # live bar chart of conditions
        c.execute('SELECT label, COUNT(*) as cnt FROM results GROUP BY label')
        label_counts = c.fetchall()
        if label_counts:
            df_labels = pd.DataFrame(label_counts)
            df_labels = df_labels.set_index('label')
            st.subheader("Conditions (counts)")
            st.bar_chart(df_labels)
        else:
            st.info("No analysis results available yet to display chart.")

        st.subheader("Recent analyses")
        c.execute('''SELECT r.id, u.username as user, r.label, r.timestamp, r.text FROM results r LEFT JOIN users u ON r.user_id = u.id ORDER BY r.timestamp DESC LIMIT 50''')
        recent = c.fetchall()
        if recent:
            df_recent = pd.DataFrame(recent)
            st.dataframe(df_recent)
            # allow admin to download full results
            df_full = pd.read_sql_query('SELECT r.id, u.username as user, r.label, r.timestamp, r.text FROM results r LEFT JOIN users u ON r.user_id = u.id ORDER BY r.timestamp DESC', conn)
            st.download_button('Download all results as CSV', df_full.to_csv(index=False).encode('utf-8'), file_name='all_results.csv')
            st.download_button('Download all results as JSON', df_full.to_json(orient='records'), file_name='all_results.json')
        else:
            st.info('No recent analyses to show yet.')

        # simple admin actions
        st.markdown('---')
        st.subheader('Admin actions')
        if st.button('Clear anonymous results older than 90 days'):
            cutoff = (datetime.utcnow() - timedelta(days=90)).isoformat()
            c.execute('DELETE FROM results WHERE user_id IS NULL AND timestamp < ?', (cutoff,))
            deleted = conn.total_changes
            conn.commit()
            st.success(f'Cleared anonymous results older than 90 days.')
        if st.button('Export users list'):
            df_users = pd.read_sql_query('SELECT id, username, is_admin, created_at FROM users', conn)
            st.download_button('Download users as CSV', df_users.to_csv(index=False).encode('utf-8'), file_name='users.csv')

        conn.close()

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Settings
# ---------------------------
elif menu == "Settings":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Settings")
    if not st.session_state['is_admin']:
        st.info("Some settings are available to all users; admin-only settings are below.")

    # site name
    site_name = st.text_input('Site name', value=get_setting('site_name', SITE_NAME_DEFAULT) or SITE_NAME_DEFAULT)
    if st.button('Save site name'):
        set_setting('site_name', site_name)
        st.success('Site name saved.')

    # chart refresh interval (stored but not enforced)
    refresh_val = int(get_setting('chart_refresh_seconds', str(DEFAULT_REFRESH_SECONDS))) if get_setting('chart_refresh_seconds') else DEFAULT_REFRESH_SECONDS
    refresh_seconds = st.number_input('Chart refresh interval (seconds)', min_value=5, max_value=3600, value=refresh_val)
    if st.button('Save refresh interval'):
        set_setting('chart_refresh_seconds', str(refresh_seconds))
        st.success('Refresh interval saved.')

    # data retention (admin only)
    if st.session_state['is_admin']:
        retention_days = int(get_setting('data_retention_days', '365'))
        retention_days = st.number_input('Data retention (days)', min_value=1, max_value=3650, value=retention_days)
        if st.button('Save retention'):
            set_setting('data_retention_days', str(retention_days))
            st.success('Data retention saved.')

        if st.button('Delete results older than retention window'):
            cutoff = (datetime.utcnow() - timedelta(days=int(get_setting('data_retention_days', '365')))).isoformat()
            conn = get_db_connection()
            c = conn.cursor()
            c.execute('DELETE FROM results WHERE timestamp < ?', (cutoff,))
            conn.commit()
            conn.close()
            st.success('Old results deleted per retention policy.')

    st.markdown('---')
    st.write('User preferences:')
    # example user preference (stored in session only)
    pref_show_resources = st.checkbox('Show detailed resource links after analysis', value=True)
    st.session_state['pref_show_resources'] = pref_show_resources

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# About
# ---------------------------
elif menu == "About":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("About")
    st.write("""
    Mind Lens is an informational tool that uses a text-based model to suggest mental health categories based on short text input.

    The application is intended for educational and supportive purposes and is not a substitute for professional diagnosis or therapy.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# default fallback
else:
    st.write('Select a page from the sidebar.')

# footer
st.markdown("<div class='footer'>Made with care • Mind Lens • 2025</div>", unsafe_allow_html=True)