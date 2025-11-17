import streamlit as st
import sqlite3
import hashlib
from datetime import datetime
import pandas as pd

# -----------------------------
# DATABASE INIT
# -----------------------------

def init_db():
    conn = sqlite3.connect("mindlens.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users(
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS results(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            text_input TEXT,
            prediction TEXT,
            timestamp TEXT
        )
    """)
    # Create default admin if not exists
    c.execute("SELECT * FROM users WHERE username='admin'")
    if not c.fetchone():
        c.execute("INSERT INTO users VALUES (?,?,?)", ('admin', hashlib.sha256('admin123'.encode()).hexdigest(), 'admin'))
    conn.commit()
    conn.close()

# -----------------------------
# UTILS
# -----------------------------

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_login(username, password):
    conn = sqlite3.connect("mindlens.db")
    c = conn.cursor()
    c.execute("SELECT role FROM users WHERE username=? AND password_hash=?", (username, hash_password(password)))
    res = c.fetchone()
    conn.close()
    return res[0] if res else None

# -----------------------------
# USER & ADMIN FUNCTIONS
# -----------------------------

def create_user(username, password, role='user'):
    try:
        conn = sqlite3.connect("mindlens.db")
        c = conn.cursor()
        c.execute("INSERT INTO users VALUES (?,?,?)", (username, hash_password(password), role))
        conn.commit()
        conn.close()
        return True
    except:
        return False

def save_result(username, text_input, prediction):
    conn = sqlite3.connect("mindlens.db")
    c = conn.cursor()
    c.execute("INSERT INTO results(username,text_input,prediction,timestamp) VALUES (?,?,?,?)", 
              (username, text_input, prediction, str(datetime.now())))
    conn.commit()
    conn.close()

# -----------------------------
# PAGE FUNCTIONS
# -----------------------------

def page_home():
    st.title("Mind Lens")
    st.write("Your mental health check assistant.")

    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        st.info("Please login to access the analysis page.")
        return

    text_input = st.text_area("Enter your text here")
    if st.button("Analyze"):
        if not text_input.strip():
            st.warning("Enter some text first.")
            return
        # Placeholder prediction logic (replace with your ML model)
        prediction = 'Positive' if len(text_input) % 2 == 0 else 'Negative'
        st.success(f"Prediction: {prediction}")
        save_result(st.session_state.username, text_input, prediction)
        # Option to download single result
        df = pd.DataFrame([{'Text': text_input, 'Prediction': prediction}])
        st.download_button("Download Result CSV", df.to_csv(index=False), file_name='result.csv')


def page_login(user_type='user'):
    st.subheader(f"{user_type.capitalize()} Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    if st.button(f"Login as {user_type.capitalize()}"):
        role = verify_login(username, password)
        if role:
            if user_type=='admin' and role != 'admin':
                st.error("Not authorized as admin.")
                return
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = role
            st.success("Login successful!")
        else:
            st.error("Invalid credentials.")


def page_signup():
    st.subheader("Create Account")
    username = st.text_input("Choose username")
    password = st.text_input("Choose password", type='password')
    if st.button("Sign Up"):
        if create_user(username, password):
            st.success("Account created. You can now login.")
        else:
            st.error("Username already exists.")


def page_admin():
    if 'logged_in' not in st.session_state or not st.session_state.logged_in or st.session_state.role != 'admin':
        st.warning("Admin access only.")
        return

    st.title("Admin Dashboard")
    conn = sqlite3.connect("mindlens.db")
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users")
    total_users = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM results")
    total_results = c.fetchone()[0]

    st.metric("Total Users", total_users)
    st.metric("Total Analyses", total_results)

    st.subheader("Recent Results")
    df = pd.read_sql_query("SELECT * FROM results ORDER BY timestamp DESC LIMIT 20", conn)
    st.dataframe(df)

    st.subheader("Download All Results")
    st.download_button("Download CSV", df.to_csv(index=False), file_name='all_results.csv')

    # Bar chart of predictions
    st.subheader("Prediction Distribution")
    chart_data = df['prediction'].value_counts().reset_index()
    chart_data.columns = ['Prediction','Count']
    st.bar_chart(chart_data.set_index('Prediction'))
    conn.close()


def page_about():
    st.title("About Mind Lens")
    st.write("Mind Lens is a mental health text analysis tool. Users can analyze their text and get a prediction. Admins can monitor usage and view results.")

# -----------------------------
# MAIN
# -----------------------------

init_db()
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

menu = ['Home','User Login','Admin Login','Sign Up','About']
if st.session_state.logged_in:
    if st.session_state.role=='admin':
        menu = ['Home','Admin','Logout','About']
    else:
        menu = ['Home','Logout','About']

choice = st.sidebar.radio("Menu", menu)

if choice=='Home':
    page_home()
elif choice=='User Login':
    page_login('user')
elif choice=='Admin Login':
    page_login('admin')
elif choice=='Sign Up':
    page_signup()
elif choice=='Admin':
    page_admin()
elif choice=='About':
    page_about()
elif choice=='Logout':
    st.session_state.logged_in=False
    st.session_state.username=None
    st.session_state.role=None
    st.success("Logged out.")
    st.experimental_rerun()