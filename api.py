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
    subjects.remove("date_entered")

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
    3. Markdown table of 10 best Bachelor’s colleges (with location, fees, eligibility, and NIRF 2025 rank).
    4. Markdown table of 5 best Master’s programs (if applicable).
    5. End with a motivational summary.
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
# ========== STREAMLIT FRONTEND ==============================
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

    def get_profile(uid):
        doc = db.collection("students").document(uid).get()
        return doc.to_dict() if doc.exists else None

    profile = get_profile(uid)

    # ================= FIRST TIME PROFILE CREATION ====================
    if not profile:
        st.subheader("👋 Welcome! Let’s set up your profile.")
        name = st.text_input("Enter your name:")
        class_studying = st.selectbox("Select your class:", [str(i) for i in range(8, 13)])
        subjects_input = st.text_area("Enter your subjects (comma-separated):")

        if st.button("💾 Save Profile"):
            subjects = [s.strip() for s in subjects_input.split(",") if s.strip()]
            if not name or not class_studying or not subjects:
                st.warning("⚠️ Please fill all details.")
            else:
                db.collection("students").document(uid).set({
                    "name": name,
                    "class": class_studying,
                    "subjects": subjects,
                    "questionnaire_done": False,
                    "domain": None
                })
                st.success("✅ Profile created successfully! Now complete the questionnaire.")
        st.stop()

    else:
        student_name = profile.get("name", "")
        student_class = profile.get("class", "")
        subjects = profile.get("subjects", [])
        questionnaire_done = profile.get("questionnaire_done", False)
        saved_domain = profile.get("domain", None)

        st.sidebar.success(f"Hello, {student_name}! 👋")

        # Navigation
        nav_pages = ["Profile Details"]
        if not questionnaire_done:
            nav_pages.append("Questionnaire")
        else:
            nav_pages += ["Counsel", "Previous Analysis"]

        page = st.sidebar.radio("📂 Navigate to:", nav_pages)

        # ---------------- Counsel Page -----------------
        if page == "Counsel":
            st.subheader("🧠 Enter Your Test Marks")

            test_scores = []
            tests_ref = db.collection("students").document(uid).collection("tests").stream()
            for doc in tests_ref:
                test_scores.append(doc.to_dict())

            st.write(f"Currently tracking subjects: {', '.join(subjects)}")

            test_data = {
                "class": student_class,
                "date_entered": datetime.datetime.now().strftime("%Y-%m-%d")
            }
            for sub in subjects:
                test_data[sub] = st.number_input(f"{sub} Marks", 0, 100, 0)

            if st.button("➕ Add Test"):
                db.collection("students").document(uid).collection("tests").add(test_data)
                st.success("✅ Test added successfully!")

            if test_scores:
                st.subheader("📊 Your Test History")
                df = pd.DataFrame(test_scores)
                columns_order = ["class", "date_entered"] + [sub for sub in subjects if sub in df.columns]
                df = df[columns_order]
                st.dataframe(df)

                st.subheader("🚀 Generate AI Career Guidance")
                state_name = st.text_input("Your State:")
                requirement = st.text_input("Career Interest:")

                if st.button("💡 Get Guidance"):
                    actual_domain = saved_domain or "Not Defined"

                    # ---------- MATCH CASE ----------
                    if requirement.lower() in actual_domain.lower():
                        mode = "direct"
                        prompt_style = f"The student's career interest '{requirement}' matches their assessed domain '{actual_domain}'. Give deep career guidance."

                        input_state = CounsellorState(
                            student_name=student_name,
                            test_scores=test_scores,
                            state=state_name,
                            requirement=requirement,
                            guidance_text=""
                        )
                        base_output = app.invoke(input_state)["guidance_text"]

                    # ---------- MISMATCH CASE 🔧 (UPDATED) ----------
                    else:
                        mode = "mismatch"
                        # Generate guidance for both interest and aptitude
                        input_interest = CounsellorState(
                            student_name=student_name,
                            test_scores=test_scores,
                            state=state_name,
                            requirement=requirement,
                            guidance_text=""
                        )
                        guidance_interest = app.invoke(input_interest)["guidance_text"]

                        input_aptitude = CounsellorState(
                            student_name=student_name,
                            test_scores=test_scores,
                            state=state_name,
                            requirement=actual_domain,
                            guidance_text=""
                        )
                        guidance_aptitude = app.invoke(input_aptitude)["guidance_text"]

                        base_output = f"""
                        ### ⚖️ Alignment Advice
                        While your interest in **{requirement}** is inspiring, your aptitude indicates a strong potential in **{actual_domain}**.
                        
                        ---
                        ## 🌟 Career Path 1: Your Interest — *{requirement}*
                        {guidance_interest}

                        ---
                        ## 💪 Career Path 2: Your Natural Strength — *{actual_domain}*
                        {guidance_aptitude}

                        ---
                        ### 💡 Final Recommendation
                        Combine both paths strategically by focusing on foundational skills from **{actual_domain}** while pursuing your passion for **{requirement}**. This will help you build a future that aligns with both your heart and your mind.
                        """

                    db.collection("students").document(uid).collection("guidance_history").add({
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d"),
                        "name": student_name,
                        "email": email,
                        "state": state_name,
                        "requirement": requirement,
                        "guidance_text": base_output,
                        "domain": actual_domain
                    })

                    st.markdown(base_output, unsafe_allow_html=True)

        # ---------------- Previous Analysis -----------------
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
                    with st.expander(
                        f"📄 Report {idx}: {report.get('requirement', 'Unknown')} ({report.get('timestamp', 'No date')})"
                    ):
                        st.write(f"👤 **Name:** {report.get('name', 'N/A')}")
                        st.write(f"📍 **State:** {report.get('state', 'N/A')}")
                        st.write(f"✉️ **Email:** {report.get('email', 'N/A')}")
                        st.write(f"🎯 **Assessed Domain:** {report.get('domain', 'N/A')}")
                        st.markdown(report.get("guidance_text", "_No guidance text found._"), unsafe_allow_html=True)
            else:
                st.info("No previous analyses found.")
else:
    st.warning("👋 Please log in or register to access your personalized dashboard.")
