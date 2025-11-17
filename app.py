# Mind Lens App (User + Admin Login)

import streamlit as st
import sqlite3
import hashlib
from datetime import datetime

# -----------------------------
# DATABASE
# -----------------------------

def init_db():
    conn = sqlite3.connect("mindlens.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users(
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS results(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            text_input TEXT,
            prediction TEXT,
            date TEXT
        )
    """)
    conn.commit()
    conn.close()

# -----------------------------
# UTILS
# -----------------------------

def hash_pw(password):
    return hashlib.sha256(password.encode()).hexdigest()

# -----------------------------
# AUTH SYSTEM
# -----------------------------

def create_user(username, password, role="user"):
    conn = sqlite3.connect("mindlens.db")
    c = conn.cursor()
    c.execute("INSERT INTO users VALUES (?, ?, ?)", (username, hash_pw(password), role))
    conn.commit()
    conn.close()


def login_user(username, password):
    conn = sqlite3.connect("mindlens.db")
    c = conn.cursor()
    c.execute("SELECT role FROM users WHERE username=? AND password=?", (username, hash_pw(password)))
    data = c.fetchone()
    conn.close()
    return data

# -----------------------------
# LAYOUT FUNCTIONS
# -----------------------------

def show_home():
    st.title("Mind Lens")
    st.write("Mental health check assistant.")

    st.subheader("Start Analysis")
    user_text = st.text_area("Enter your text")
    if st.button("Analyze"):
        prediction = "Positive" if len(user_text) % 2 == 0 else "Negative"

        conn = sqlite3.connect("mindlens.db")
        c = conn.cursor()
        c.execute("INSERT INTO results(username,text_input,prediction,date) VALUES(?,?,?,?)",
                  (st.session_state.username, user_text, prediction, str(datetime.now())))
        conn.commit()
        conn.close()

        st.success("Analysis complete.")
        st.write("Prediction:", prediction)


def show_admin():
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
    df = None
    try:
        import pandas as pd
        df = pd.read_sql_query("SELECT * FROM results ORDER BY id DESC LIMIT 20", conn)
        st.dataframe(df)
    except:
        st.write("Install pandas to view data.")

    conn.close()

# -----------------------------
# PAGES
# -----------------------------

def show_login(user_type="user"):
    st.title(f"{user_type.capitalize()} Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button(f"Login as {user_type.capitalize()}"):
        data = login_user(username, password)
        if data:
            if user_type == "admin" and data[0] != "admin":
                st.error("This is an admin login only.")
            else:
                st.session_state.logged = True
                st.session_state.username = username
                st.session_state.role = data[0]
                st.success("Logged in successfully.")
        else:
            st.error("Invalid credentials.")


def show_signup():
    st.title("Create Account")
    username = st.text_input("Choose username")
    password = st.text_input("Choose password", type="password")

    if st.button("Create Account"):
        try:
            create_user(username, password)
            st.success("Account created. You can now log in.")
        except:
            st.error("Username already exists.")

# -----------------------------
# MAIN APP
# -----------------------------

init_db()

if "logged" not in st.session_state:
    st.session_state.logged = False

if st.session_state.logged:
    if st.session_state.role == "admin":
        menu = ["Home", "Admin", "Logout"]
    else:
        menu = ["Home", "Logout"]
else:
    menu = ["Home", "User Login", "Admin Login", "Sign Up"]

choice = st.sidebar.radio("Navigation", menu)

if choice == "Home":
    if not st.session_state.logged:
        st.warning("Please log in to access the app.")
    else:
        show_home()

elif choice == "User Login":
    show_login(user_type="user")

elif choice == "Admin Login":
    show_login(user_type="admin")

elif choice == "Sign Up":
    show_signup()

elif choice == "Admin":
    show_admin()

elif choice == "Logout":
    st.session_state.logged = False
    st.session_state.username = None
    st.session_state.role = None
    st.success("Logged out.")
    st.experimental_rerun()