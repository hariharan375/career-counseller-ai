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
        firebase_key = json.loads(st.secrets["FIREBASE_KEY"])
        cred = credentials.Certificate(firebase_key)
    else:
        cred = credentials.Certificate(
            "career counseller ai/career-counsellor-ai-firebase-adminsdk-fbsvc-ad36c831af.json"
        )
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ============================================================
# ========== GROQ INITIALIZATION =============================
# ============================================================
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ============================================================
# ========== LANGGRAPH STATE =================================
# ============================================================
class CounsellorState(TypedDict):
    student_name: str
    test_scores: List[Dict[str, int]]
    state: str
    requirement: str
    guidance_text: str


def career_guidance_node(state: CounsellorState):
    trends = {}
    subjects = list(state["test_scores"][0].keys())
    subjects.remove("class")

    def get_trend(scores):
        if len(scores) < 2:
            return "Not enough data"
        if scores[-1] > scores[0]:
            return "Improving"
        elif scores[-1] < scores[0]:
            return "Declining"
        else:
            return "Stable"

    for sub in subjects:
        sub_scores = [t[sub] for t in state["test_scores"] if sub in t]
        trends[sub] = get_trend(sub_scores)

    prompt = f"""
    The student named {state['student_name']} has these academic details:
    - Test Scores: {state['test_scores']}
    - Trends: {trends}
    - State: {state['state']}
    - Career Interest: {state['requirement']}

    Provide:
    1. Personalized career guidance with improvement areas and clear action steps.
    2. Mention each subject's trend and what it means.
    3. Markdown table of 10 best **Bachelor’s** colleges (their location and avg fees as columns) related to their interest according to the marks and capability nearby their given location, preferably along with their eligibility criteria and NIRF ranking 2025 in separate columns.
    4. Markdown table of 5 best **Master’s** programs (if applicable).
    5. End with a brief summary about feasible career options and their capabilities for the same, followed by a motivational note.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )
    output = response.choices[0].message.content.strip()
    return {**state, "guidance_text": output}


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

email = st.sidebar.text_input("Email")
password = st.sidebar.text_input("Password", type="password")

if "user" not in st.session_state:
    st.session_state.user = None

# Register
if auth_mode == "Register" and st.sidebar.button("Create Account"):
    try:
        user = auth.create_user(email=email, password=password)
        st.sidebar.success("✅ Account created! Please login.")
    except Exception as e:
        st.sidebar.error(f"⚠️ Error: {e}")

# Login
if auth_mode == "Login" and st.sidebar.button("Login"):
    try:
        user = auth.get_user_by_email(email)
        st.session_state.user = user
        st.sidebar.success(f"✅ Welcome {email}")
    except Exception as e:
        st.sidebar.error(f"⚠️ Login failed: {e}")

# Logout
if st.session_state.user:
    if st.sidebar.button("🚪 Logout"):
        st.session_state.user = None
        st.session_state.clear()
        st.success("✅ Logged out successfully.")
        st.stop()

# ------------------------------------------------------------
# 🧠 MAIN APP SECTION
# ------------------------------------------------------------
user = st.session_state.user
if user:
    uid = user.uid

    # Load or create profile
    def get_profile(uid):
        doc = db.collection("students").document(uid).get()
        return doc.to_dict() if doc.exists else None

    profile = get_profile(uid)

    if not profile:
        st.subheader("👋 Welcome! Let’s set up your profile.")
        name = st.text_input("Enter your name:")
        class_studying = st.selectbox("Select your class:", [str(i) for i in range(8, 13)])
        subjects_input = st.text_area(
            "Enter your subjects separated by commas (e.g., Physics, Chemistry, Maths):"
        )

        if st.button("💾 Save Profile"):
            subjects = [s.strip() for s in subjects_input.split(",") if s.strip()]
            db.collection("students").document(uid).set(
                {"name": name, "class": class_studying, "subjects": subjects}
            )
            st.success("✅ Profile created! Please refresh to continue.")
            st.stop()

    else:
        # Profile exists
        student_name = profile.get("name", "")
        student_class = profile.get("class", "")
        subjects = profile.get("subjects", [])
        st.sidebar.success(f"Hello, {student_name}! 👋")

        # Navigation
        page = st.sidebar.radio("📂 Navigate to:", ["Profile Details", "Counsel", "Previous Analysis"])

        # --------------------------------------------------------
        # 🧾 PROFILE DETAILS PAGE
        # --------------------------------------------------------
        if page == "Profile Details":
            st.subheader("📋 Profile Details")

            new_name = st.text_input("Your Name", value=student_name)
            new_class = st.selectbox("Your Class", [str(i) for i in range(8, 13)], index=int(student_class) - 8)
            subjects_str = ", ".join(subjects)
            new_subjects = st.text_area("Subjects (comma-separated):", value=subjects_str)

            if st.button("💾 Update Profile"):
                updated_subjects = [s.strip() for s in new_subjects.split(",") if s.strip()]
                db.collection("students").document(uid).update(
                    {"name": new_name, "class": new_class, "subjects": updated_subjects}
                )
                st.success("✅ Profile updated! Please refresh.")
                st.stop()

        # --------------------------------------------------------
        # 🎯 COUNSEL PAGE
        # --------------------------------------------------------
        elif page == "Counsel":
            st.subheader("🧠 Enter Your Test Marks")

            test_scores = []
            tests_ref = db.collection("students").document(uid).collection("tests").stream()
            for doc in tests_ref:
                test_scores.append(doc.to_dict())

            st.write(f"Currently tracking subjects: {', '.join(subjects)}")

            test_data = {"class": student_class}
            for sub in subjects:
                test_data[sub] = st.number_input(f"{sub} Marks", 0, 100, 0)

            if st.button("➕ Add Test"):
                db.collection("students").document(uid).collection("tests").add(test_data)
                st.success("✅ Test added successfully! Please refresh to view.")

            if test_scores:
                st.subheader("📊 Your Test History")

                df = pd.DataFrame(test_scores)
                for col in ["timestamp", "id"]:
                    if col in df.columns:
                        df = df.drop(columns=[col])

                # Reorder columns → Class first, then subjects
                columns_order = ["class"] + [sub for sub in subjects if sub in df.columns]
                other_columns = [c for c in df.columns if c not in columns_order]
                df = df[columns_order + other_columns]

                st.dataframe(
                    df.style.set_properties(**{'text-align': 'center'}).set_table_styles([
                        {'selector': 'th', 'props': [('text-align', 'center')]}
                    ])
                )

                st.subheader("🚀 Generate AI Career Guidance")
                state_name = st.text_input("Your State (e.g., Tamil Nadu)")
                requirement = st.text_input("Career Interest (e.g., Engineering, Medicine, Design)")

                if st.button("💡 Get Guidance"):
                    input_state = CounsellorState(
                        student_name=student_name,
                        test_scores=test_scores,
                        state=state_name,
                        requirement=requirement,
                        guidance_text=""
                    )
                    final_state = app.invoke(input_state)

                    db.collection("students").document(uid).collection("guidance_history").add({
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d"),
                        "name": student_name,
                        "email": email,
                        "state": state_name,
                        "requirement": requirement,
                        "guidance_text": final_state["guidance_text"]
                    })

                    st.markdown(final_state["guidance_text"], unsafe_allow_html=True)

        # --------------------------------------------------------
        # 🕒 PREVIOUS ANALYSIS PAGE
        # --------------------------------------------------------
        elif page == "Previous Analysis":
            st.subheader("🕒 Previous Career Guidance Reports")

            reports_ref = (
                db.collection("students")
                .document(uid)
                .collection("guidance_history")
                .order_by("timestamp", direction=firestore.Query.DESCENDING)
                .stream()
            )
            reports = [doc.to_dict() for doc in reports_ref]

            if reports:
                for idx, report in enumerate(reports, start=1):
                    with st.expander(f"📄 Report {idx}: {report.get('requirement', 'Unknown Interest')} ({report.get('timestamp', 'No date')})"):
                        st.write(f"👤 **Name:** {report.get('name', 'N/A')}")
                        st.write(f"📍 **State:** {report.get('state', 'N/A')}")
                        st.write(f"✉️ **Email:** {report.get('email', 'N/A')}")
                        st.markdown("---")
                        st.markdown(report.get("guidance_text", "_No guidance text found._"), unsafe_allow_html=True)
            else:
                st.info("No previous analyses found.")
else:
    st.warning("👋 Please log in or register to access your personalized dashboard.")
