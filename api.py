import streamlit as st
import pandas as pd
import numpy as np
import json
import hashlib
import firebase_admin
from firebase_admin import credentials, firestore, auth

# ========== FIREBASE INITIALIZATION ==========
if not firebase_admin._apps:
    # Use Firebase credentials from Streamlit Secrets (for deployment)
    if "FIREBASE_KEY" in st.secrets:
        firebase_key = json.loads(st.secrets["FIREBASE_KEY"])
        cred = credentials.Certificate(firebase_key)
    else:
        # For local testing, use your downloaded serviceAccountKey.json
        cred = credentials.Certificate("serviceAccountKey.json")

    firebase_admin.initialize_app(cred)

db = firestore.client()

# ========== HELPER FUNCTIONS ==========
def make_hash(password):
    """Basic SHA256 hashing for demo purposes."""
    return hashlib.sha256(password.encode()).hexdigest()

def analyze_trend(test_scores):
    """Determine performance trend based on average marks per test."""
    if len(test_scores) < 2:
        return "Not enough data to determine trend."
    averages = [np.mean(list(t.values())) for t in test_scores]
    if averages[-1] > averages[0]:
        return "Your performance is improving 📈 — keep it up!"
    elif averages[-1] < averages[0]:
        return "Your performance has decreased 📉 — focus more on weak areas."
    else:
        return "Your performance is stable ⚖️ — try to push for consistent growth."

def career_guidance_based_on_marks(avg):
    """Simple career suggestions based on score averages."""
    if avg >= 90:
        return "Excellent! You could explore careers in Research, Data Science, or Engineering."
    elif avg >= 75:
        return "Great job! Consider careers in Software, Electronics, or Management fields."
    elif avg >= 60:
        return "Good effort! You might like roles in Design, Technical Support, or Analytics."
    else:
        return "Don’t give up! Focus on improving concepts and seek guidance. You can explore vocational or creative fields."

# ========== STREAMLIT APP UI ==========
st.set_page_config(page_title="AI Career Counsellor", layout="centered")
st.title("🎓 AI Career Counsellor with Progress Tracking")
st.caption("An AI-powered system for personalized career guidance and academic progress analysis.")

# -------- SIDEBAR AUTH --------
st.sidebar.title("🔑 User Authentication")
auth_mode = st.sidebar.radio("Choose Action:", ["Login", "Register"])
email = st.sidebar.text_input("Email")
password = st.sidebar.text_input("Password", type="password")

user = None

if auth_mode == "Register" and st.sidebar.button("Create Account"):
    try:
        user = auth.create_user(email=email, password=password)
        st.sidebar.success("✅ Account created! Please login.")
    except Exception as e:
        st.sidebar.error(f"⚠️ Error: {e}")

if auth_mode == "Login" and st.sidebar.button("Login"):
    try:
        user = auth.get_user_by_email(email)
        st.sidebar.success(f"✅ Welcome {email}")
        st.session_state["user"] = user
    except Exception as e:
        st.sidebar.error(f"⚠️ Login failed: {e}")

# -------- MAIN PAGE --------
if "user" in st.session_state:
    user = st.session_state["user"]
    st.success(f"Logged in as: {email}")

    # Load previous test data
    st.session_state.test_scores = []
    tests_ref = db.collection("students").document(user.uid).collection("tests").stream()
    for doc in tests_ref:
        st.session_state.test_scores.append(doc.to_dict())

    st.subheader("📚 Enter New Test Marks")

    physics = st.number_input("Physics Marks", 0, 100, 0)
    chemistry = st.number_input("Chemistry Marks", 0, 100, 0)
    maths = st.number_input("Maths Marks", 0, 100, 0)

    if st.button("Add Test"):
        test_data = {
            "Physics": physics,
            "Chemistry": chemistry,
            "Maths": maths
        }

        # Save to Firestore
        db.collection("students").document(user.uid).collection("tests").add(test_data)
        st.session_state.test_scores.append(test_data)

        st.success("✅ Test data added successfully!")

    # -------- DISPLAY PREVIOUS TESTS --------
    if st.session_state.test_scores:
        st.subheader("📊 Your Test History")
        df = pd.DataFrame(st.session_state.test_scores)
        st.dataframe(df)

        avg_marks = df.mean(axis=1)
        overall_avg = df.mean().mean()

        st.write(f"**Overall Average Marks:** {overall_avg:.2f}")
        st.write(analyze_trend(st.session_state.test_scores))

        # Career guidance output
        st.subheader("💡 Personalized Career Guidance")
        suggestion = career_guidance_based_on_marks(overall_avg)
        st.success(suggestion)
    else:
        st.info("No tests added yet. Start by entering your first test data below.")

else:
    st.warning("👋 Please log in or register to access your personalized dashboard.")
