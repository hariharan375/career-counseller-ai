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
            db.collection("students").document(uid).set({
                "name": name,
                "class": class_studying,
                "subjects": subjects,
                "questionnaire_done": False
            })
            st.success("✅ Profile created! Please refresh to continue.")
            st.stop()

    else:
        student_name = profile.get("name", "")
        student_class = profile.get("class", "")
        subjects = profile.get("subjects", [])
        questionnaire_done = profile.get("questionnaire_done", False)

        st.sidebar.success(f"Hello, {student_name}! 👋")
        nav_pages = ["Profile Details", "Counsel", "Previous Analysis"]
        if not questionnaire_done:
            nav_pages.insert(1, "Questionnaire")

        page = st.sidebar.radio("📂 Navigate to:", nav_pages)

        # ---------------- Profile Page -----------------
        if page == "Profile Details":
            st.subheader("📋 Profile Details")

            new_name = st.text_input("Your Name", value=student_name)
            new_class = st.selectbox("Your Class", [str(i) for i in range(8, 13)],
                                     index=int(student_class) - 8)
            subjects_str = ", ".join(subjects)
            new_subjects = st.text_area("Subjects (comma-separated):", value=subjects_str)

            if st.button("💾 Update Profile"):
                updated_subjects = [s.strip() for s in new_subjects.split(",") if s.strip()]
                db.collection("students").document(uid).update({
                    "name": new_name, "class": new_class, "subjects": updated_subjects
                })
                st.success("✅ Profile updated! Refresh to see changes.")

        # ---------------- Questionnaire Page -----------------
        elif page == "Questionnaire":
            st.subheader("🧩 Career Interest Questionnaire")
            st.info("Please rate each statement from 1 (Strongly Disagree) to 5 (Strongly Agree). You can submit only once.")

            questions = [
                "1. I love understanding how engines, bikes, and machines work and imagining how to make them faster or more efficient.",
                "2. I am interested in learning how rockets work and how humans explore space and other planets.",
                "3. The idea of building or programming a robot that can move or think on its own excites me.",
                "4. I am interested in studying the human body and understanding how diseases are diagnosed and treated.",
                "5. I am curious to understand how a human mind works and how people react in different situations.",
                "6. I am curious about maintaining law and order and serving the public as part of the police services.",
                "7. I like creating visuals, designs, or media that express ideas and information clearly.",
                "8. I get curious about how gadgets, circuits, and electrical systems power our homes and devices.",
                "9. I am interested in exploring how natural herbs and traditional healing methods help maintain good health.",
                "10. I enjoy imagining and designing buildings, spaces, and structures that are both functional and beautiful.",
                "11. I’m fascinated by how bridges, buildings, or industries are designed to be safe, efficient, and sustainable.",
                "12. I enjoy solving real-life problems using physics and mathematical concepts.",
                "13. I am curious about how medicines work in the body and how physical therapy helps in recovery.",
                "14. I am interested in designing clothes, following fashion trends, and creating my own style ideas.",
                "15. I am interested in understanding trade, business transactions, and how markets operate.",
                "16. I am interested in learning about laws, legal systems, and how justice is delivered.",
                "17. I enjoy learning how hotels, restaurants, and tourism services are managed to provide great experiences.",
                "18. I would like to pursue my favorite sport as a professional career.",
                "19. I like learning about managing organizations, planning work, and improving business operations.",
                "20. I like learning about society, cultures, and how communities interact and develop.",
                "21. I would like to learn how to care for teeth, gums, and overall oral health.",
                "22. I enjoy solving problems using computers and want to learn how apps, games, or AI tools are created.",
                "23. I am curious about how living organisms can be used to develop medicines, improve crops, or solve health problems.",
                "24. I want to learn how to treat and take care of animals and understand their health conditions.",
                "25. I would like to join NDA to receive joint training for the army, navy, or air force to defend the country.",
                "26. I want to understand how science and technology help in improving farming and food production.",
                "27. I am curious about accounting, auditing, and how financial decisions are made in companies.",
                "28. I would like to work in government administration and contribute to policy-making and governance.",
                "29. I enjoy reading, writing, and understanding stories, poetry, or different languages.",
                "30. I am interested in how money, investments, and financial planning work in businesses and daily life.",
                "31. I am interested in representing my country abroad and working in international relations and diplomacy."
            ]

            responses = {}
            for i, q in enumerate(questions, 1):
                responses[f"Q{i}"] = st.slider(q, 1, 5, 3)

            if st.button("✅ Submit Responses"):
                db.collection("students").document(uid).collection("questionnaire").add({
                    "responses": responses,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d")
                })
                db.collection("students").document(uid).update({"questionnaire_done": True})
                st.success("🎯 Questionnaire submitted successfully! You can’t edit it later.")
                st.stop()

        # ---------------- Counsel Page -----------------
        elif page == "Counsel":
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
                st.success("✅ Test added successfully! Refresh to view.")

            if test_scores:
                st.subheader("📊 Your Test History")
                df = pd.DataFrame(test_scores)
                for col in ["id"]:
                    if col in df.columns:
                        df = df.drop(columns=[col])

                columns_order = ["class", "date_entered"] + [sub for sub in subjects if sub in df.columns]
                other_columns = [c for c in df.columns if c not in columns_order]
                df = df[columns_order + other_columns]

                st.dataframe(df)

                st.subheader("🚀 Generate AI Career Guidance")
                state_name = st.text_input("Your State:")
                requirement = st.text_input("Career Interest:")

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
                        st.markdown(report.get("guidance_text", "_No guidance text found._"), unsafe_allow_html=True)
            else:
                st.info("No previous analyses found.")
else:
    st.warning("👋 Please log in or register to access your personalized dashboard.")

