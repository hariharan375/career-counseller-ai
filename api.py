import streamlit as st
import pandas as pd
import json
import firebase_admin
from firebase_admin import credentials, firestore, auth
from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, List
from groq import Groq
import datetime

# ============================================================
# ========== FIREBASE INITIALIZATION =========================
# ============================================================
if not firebase_admin._apps:
    if "FIREBASE_KEY" in st.secrets:
        try:
            firebase_key = json.loads(st.secrets["FIREBASE_KEY"])
            cred = credentials.Certificate(firebase_key)
        except Exception as e:
            st.error(f"⚠️ Failed to load Firebase key from secrets: {e}")
            st.stop()
    else:
        cred = credentials.Certificate(
            "career counseller ai/career-counsellor-ai-firebase-adminsdk-fbsvc-ad36c831af.json"
        )

    firebase_admin.initialize_app(cred)

# Firestore DB instance
db = firestore.client()

# ============================================================
# ========== GROQ INITIALIZATION ==============================
# ============================================================
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    st.error(f"⚠️ Failed to initialize Groq client: {e}")
    st.stop()

# ============================================================
# ========== LANGGRAPH STATE DEFINITION =======================
# ============================================================
class CounsellorState(TypedDict):
    student_name: str
    test_scores: List[Dict[str, int]]
    state: str
    requirement: str
    guidance_text: str

# ============================================================
# ========== LANGGRAPH NODE FUNCTION ==========================
# ============================================================
def career_guidance_node(state: CounsellorState):
    subjects = list(state["test_scores"][0].keys()) if state["test_scores"] else []
    trends = {}

    def get_trend(scores):
        if len(scores) < 2:
            return "Not enough data"
        if scores[-1] > scores[0]:
            return "Improving"
        elif scores[-1] < scores[0]:
            return "Declining"
        else:
            return "Stable"

    for subject in subjects:
        subject_scores = [t[subject] for t in state["test_scores"] if subject in t]
        trends[subject] = get_trend(subject_scores)

    prompt = f"""
    The student named {state['student_name']} has these academic details:
    - Test Scores: {state['test_scores']}
    - Trends: {trends}
    - State: {state['state']}
    - Career Interest: {state['requirement']}

    Provide:
    1. Personalized career guidance with improvement areas and clear action steps.
    2. Mention each subject's trend and what it means.
    3. Markdown table of 10 best **Bachelor’s** colleges related to their interest according to the marks and capability nearby their given location, preferably along with their eligibility criteria and NIRF ranking 2025 in separate columns.
    4. Markdown table of 5 best **Master’s** programs (if applicable).
    5. End with a brief summary about feasible career options and a motivational note.
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
        )
        output = response.choices[0].message.content.strip()
        return {**state, "guidance_text": output}
    except Exception as e:
        return {"guidance_text": f"⚠️ Error fetching AI guidance: {str(e)}"}

# ============================================================
# ========== BUILD LANGGRAPH ================================
# ============================================================
graph = StateGraph(CounsellorState)
graph.add_node("career_guidance", career_guidance_node)
graph.set_entry_point("career_guidance")
graph.add_edge("career_guidance", END)
app = graph.compile()

# ============================================================
# ========== STREAMLIT FRONTEND ===============================
# ============================================================
st.set_page_config(page_title="AI Career Counsellor", layout="wide")
st.title("🎓 AI Enabled Career Assistance")

# ------------------------------------------------------------
# 🔐 SIDEBAR: Authentication
# ------------------------------------------------------------
st.sidebar.title("🔑 User Authentication")
auth_mode = st.sidebar.radio("Choose Action:", ["Login", "Register"])

email = st.sidebar.text_input("Username (email)")
password = st.sidebar.text_input("Password", type="password")

if "user" not in st.session_state:
    st.session_state.user = None

if auth_mode == "Register" and st.sidebar.button("Create Account"):
    try:
        user = auth.create_user(email=email, password=password)
        st.sidebar.success("✅ Account created! Please login.")
    except Exception as e:
        st.sidebar.error(f"⚠️ Error: {e}")

if auth_mode == "Login" and st.sidebar.button("Login"):
    try:
        user = auth.get_user_by_email(email)
        st.session_state.user = user
        st.sidebar.success(f"✅ Welcome {email}")
    except Exception as e:
        st.sidebar.error(f"⚠️ Login failed: {e}")

if st.session_state.user:
    if st.sidebar.button("🚪 Logout"):
        st.session_state.user = None
        st.session_state.clear()
        st.success("✅ Logged out successfully.")
        st.stop()

user = st.session_state.user
if not user:
    st.warning("👋 Please log in or register to access your personalized dashboard.")
    st.stop()

uid = user.uid
st.sidebar.success(f"Logged in as: {email}")

# ------------------------------------------------------------
# 👤 FETCH OR CREATE USER PROFILE
# ------------------------------------------------------------
def get_profile(uid):
    doc = db.collection("students").document(uid).get()
    return doc.to_dict() if doc.exists else None

profile = get_profile(uid)

if not profile:
    st.subheader("🧾 Complete Your Profile")
    name = st.text_input("Enter your name")
    student_class = st.selectbox("Select your class/grade", [str(i) for i in range(8, 13)])
    subjects_input = st.text_area("Enter subjects to track (comma-separated)", placeholder="e.g., Physics, Chemistry, Maths")

    if st.button("💾 Save Profile"):
        subjects = [s.strip() for s in subjects_input.split(",") if s.strip()]
        db.collection("students").document(uid).set({
            "name": name,
            "class": student_class,
            "subjects": subjects,
            "created_at": datetime.date.today().strftime("%Y-%m-%d"),
        })
        st.success("✅ Profile saved! Please refresh to continue.")
        st.stop()
else:
    student_name = profile.get("name", "")
    student_class_profile = profile.get("class", "")
    subjects = profile.get("subjects", [])
    st.success(f"👋 Hello {student_name} (Class {student_class_profile})")

    # ========================================================
    # MAIN NAVIGATION
    # ========================================================
    page = st.sidebar.radio("📂 Navigate to:", ["Profile", "Counsel", "Previous Analysis"])

    # ========================================================
    # PROFILE PAGE
    # ========================================================
    if page == "Profile":
        st.subheader("👤 Profile Details")
        st.write(f"**Name:** {student_name}")
        st.write(f"**Class:** {student_class_profile}")
        st.write(f"**Subjects:** {', '.join(subjects)}")

        st.markdown("---")
        st.subheader("✏️ Edit Profile")
        new_name = st.text_input("Edit Name", student_name)
        new_class = st.selectbox("Edit Class", [str(i) for i in range(8, 13)], index=[str(i) for i in range(8, 13)].index(student_class_profile))
        new_subjects = st.text_area("Edit Subjects (comma separated):", ", ".join(subjects))

        if st.button("💾 Update Profile"):
            updated_subjects = [s.strip() for s in new_subjects.split(",") if s.strip()]
            db.collection("students").document(uid).update({
                "name": new_name,
                "class": new_class,
                "subjects": updated_subjects,
            })
            st.success("✅ Profile updated! Please refresh.")
            st.stop()

    # ========================================================
    # COUNSEL PAGE (ENTER MARKS + AI GUIDANCE)
    # ========================================================
    elif page == "Counsel":
        st.subheader("📚 Enter New Test Marks")
        test_scores = []

        tests_ref = db.collection("students").document(uid).collection("tests").stream()
        for doc in tests_ref:
            test_scores.append(doc.to_dict())

        inputs = {}
        for sub in subjects:
            inputs[sub] = st.number_input(f"{sub} Marks", 0, 100, 0)

        if st.button("➕ Add Test"):
            test_data = {"class": student_class_profile}
            test_data.update(inputs)
            db.collection("students").document(uid).collection("tests").add(test_data)
            st.success("✅ Test data added successfully! Please refresh to view updated list.")

        if test_scores:
            st.subheader("📊 Your Test History")
            df = pd.DataFrame(test_scores)
            if "timestamp" in df.columns:
                df = df.drop(columns=["timestamp"])
            st.dataframe(df)

            st.subheader("🧠 Get Personalized AI Career Guidance")
            state_name = st.text_input("Your State (e.g., Tamil Nadu)")
            requirement = st.text_input("Career Interest (e.g., Engineering, Medicine, Design)")

            if st.button("🚀 Generate Guidance"):
                input_state = CounsellorState(
                    student_name=student_name,
                    test_scores=test_scores,
                    state=state_name,
                    requirement=requirement,
                    guidance_text=""
                )
                final_state = app.invoke(input_state)

                db.collection("students").document(uid).collection("guidance_history").add({
                    "date": datetime.date.today().strftime("%Y-%m-%d"),
                    "name": student_name,
                    "email": email,
                    "state": state_name,
                    "requirement": requirement,
                    "guidance_text": final_state["guidance_text"]
                })

                st.subheader("📌 AI Career Guidance")
                st.markdown(final_state["guidance_text"], unsafe_allow_html=True)
        else:
            st.info("No tests added yet. Start by entering your first test data below.")

    # ========================================================
    # PREVIOUS ANALYSIS PAGE
    # ========================================================
    elif page == "Previous Analysis":
        st.subheader("🕒 Your Previous Career Guidance Reports")
        reports_ref = (
            db.collection("students")
            .document(uid)
            .collection("guidance_history")
            .order_by("date", direction=firestore.Query.DESCENDING)
            .stream()
        )
        reports = [doc.to_dict() for doc in reports_ref]

        if reports:
            for idx, report in enumerate(reports, start=1):
                with st.expander(f"📄 Report {idx}: {report.get('requirement', 'Unknown')} ({report.get('date', 'No date')})"):
                    st.write(f"👤 **Name:** {report.get('name', 'N/A')}")
                    st.write(f"📍 **State:** {report.get('state', 'N/A')}")
                    st.write(f"✉️ **Email:** {report.get('email', 'N/A')}")
                    st.markdown("---")
                    st.markdown(report.get("guidance_text", "_No guidance text found._"), unsafe_allow_html=True)
        else:
            st.info("No previous analyses found. Generate your first guidance from the Counsel page.")
