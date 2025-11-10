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
    3. Markdown table of 10 best Bachelor’s degree colleges nearer to the location given by the user only 
       (with location, program mentioned, entrance and eligibility, speciality of each college).
    4. Markdown table of 5 best Master’s programs (if applicable, with the same specifications given for the bachelors table).
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

if "clear_fields" not in st.session_state:
    st.session_state.clear_fields = False
if "user" not in st.session_state:
    st.session_state.user = None

st.sidebar.title("🔑 User Authentication")
auth_mode = st.sidebar.radio("Choose Action:", ["Login", "Register"])

# If trigger set, set widget values to blank then unset the trigger (just once per rerun!)
if st.session_state.clear_fields:
    st.session_state.email_input = ""
    st.session_state.password_input = ""
    st.session_state.clear_fields = False

email = st.sidebar.text_input("Email", key="email_input")
password = st.sidebar.text_input("Password", type="password", key="password_input")

# Register
if auth_mode == "Register" and st.sidebar.button("Create Account"):
    try:
        user = auth.create_user(email=email, password=password)
        st.sidebar.success("✅ Account created! Please login.")
        st.session_state.clear_fields = True
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"⚠️ Error: {e}")

# Login
if auth_mode == "Login" and st.sidebar.button("Login"):
    try:
        user = auth.get_user_by_email(email)
        st.session_state.user = user
        st.sidebar.success(f"✅ Welcome {email}")
        st.session_state.clear_fields = True
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"⚠️ Login failed: {e}")


# Logout
if st.session_state.user:
    if st.sidebar.button("🚪 Logout"):
        st.session_state.user = None
        st.session_state.clear()
        st.success("✅ Logged out successfully.")
        st.rerun()

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

        # ---------------- Profile Page -----------------
        if page == "Profile Details":
            st.subheader("📋 Profile Details")
            new_name = st.text_input("Your Name", value=student_name)
            new_class = st.selectbox("Your Class", [str(i) for i in range(8, 13)], index=int(student_class) - 8)
            subjects_str = ", ".join(subjects)
            new_subjects = st.text_area("Subjects (comma-separated):", value=subjects_str)

            if st.button("💾 Update Profile"):
                updated_subjects = [s.strip() for s in new_subjects.split(",") if s.strip()]
                db.collection("students").document(uid).update({
                    "name": new_name,
                    "class": new_class,
                    "subjects": updated_subjects
                })
                st.success("✅ Profile updated successfully!")

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
                "31. I am interested in representing my country abroad and working in international relations and diplomacy."
            ]

            responses = {}
            for i, q in enumerate(questions, 1):
                responses[f"Q{i}"] = st.slider(q, 1, 5, 3)

            if st.button("✅ Submit Responses"):
                domain_map = {
                    "Engineering & Technology": [1, 3, 8, 11, 22],
                    "Research & Science": [2, 7, 12, 23, 26],
                    "Medical & Life Sciences": [4, 9, 13, 21, 24],
                    "Arts & Design": [5, 10, 14, 20, 29],
                    "Business & Management": [15, 17, 19, 27, 30],
                    "Law & Public Services": [6, 16, 25, 28, 31],
                }

                domain_scores = {d: sum(responses[f"Q{i}"] for i in qn) for d, qn in domain_map.items()}
                top_domain = max(domain_scores, key=domain_scores.get)

                db.collection("students").document(uid).collection("questionnaire").add({
                    "responses": responses,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d"),
                    "calculated_domain": top_domain
                })

                db.collection("students").document(uid).update({
                    "questionnaire_done": True,
                    "domain": top_domain
                })

                st.success("🎯 Questionnaire submitted successfully! You can now access the full dashboard.")
                st.rerun()
        # ---------------- Counsel Page with Dynamic Questions -----------------
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
                st.success("✅ Test added successfully!")

            if test_scores:
                st.subheader("📊 Your Test History")
                df = pd.DataFrame(test_scores)
                columns_order = ["class", "date_entered"] + [sub for sub in subjects if sub in df.columns]
                df = df[columns_order]
                st.dataframe(df)

            st.subheader("🚀 Career Guidance Assistant")
            state_name = st.text_input("Your State:")
            requirement = st.text_input("Career Interest (e.g., Doctor, Engineer, Designer):")

            if requirement:
                st.info(f"Let's understand your interest in **{requirement}** better.")

                followup_qs = {
                    "engineer": [
                        "Which engineering fields interest you most (Mechanical, Computer, Electrical, etc.)?",
                        "Do you enjoy practical problem-solving or software-based creativity?",
                        "What kind of projects or innovations inspire you?"
                    ],
                    "doctor": [
                        "What made you interested in becoming a doctor?",
                        "Are you more inclined towards research, patient care, or surgery?",
                        "Which subjects fascinate you most — biology, chemistry, or something else?"
                    ],
                    "designer": [
                        "What type of design excites you most (fashion, architecture, graphics)?",
                        "Do you prefer visual creativity or structural design?",
                        "What do you want your designs to express or impact?"
                    ],
                    "lawyer": [
                        "What part of law fascinates you — justice, debate, or policymaking?",
                        "Would you prefer working in civil, criminal, or corporate law?",
                        "How do you view fairness and justice in society?"
                    ],
                    "scientist": [
                        "Which area of science do you like most — physics, biology, or chemistry?",
                        "Do you enjoy experimentation or theoretical research?",
                        "Would you like to work in labs, universities, or industries?"
                    ],
                }

                key = next((k for k in followup_qs.keys() if k in requirement.lower()), None)
                if key:
                    qs = followup_qs[key]
                else:
                    qs = [
                        "What draws you toward this field?",
                        "What kind of daily work or challenges excite you in this area?",
                        "Where do you see yourself applying these skills in the future?"
                    ]

                answers = []
                for q in qs:
                    answers.append(st.text_area(q))

                if all(answers) and st.button("💡 Generate Career Guidance"):
                    narrow_prompt = f"""
                    The student {student_name} expressed interest in {requirement} and answered:
                    {dict(zip(qs, answers))}.
                    Based on these, identify their most fitting specialization or sub-field.
                    Provide a proper reasoning.
                    """

                    subresp = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": narrow_prompt}],
                    )
                    specialization = subresp.choices[0].message.content.strip()

                    actual_domain = saved_domain or "Not Defined"
                    matched = requirement.lower() in actual_domain.lower()
                    mode = "direct" if matched else "mismatch"

                    input_state = CounsellorState(
                        student_name=student_name,
                        test_scores=test_scores,
                        state=state_name,
                        requirement=f"{requirement} - {specialization}",
                        guidance_text=""
                    )

                    base_output = app.invoke(input_state)["guidance_text"]

                    if mode == "mismatch":
                        base_output = (
                            f"### ⚖️ Alignment Advice\n"
                            f"Your passion for **{requirement}** (specialized in *{specialization}*) is great, "
                            f"but your aptitude test indicated strength in **{actual_domain}**.\n\n"
                            f"Here's how you can align both fields effectively:\n\n"
                            f"{base_output}"
                        )

                    db.collection("students").document(uid).collection("guidance_history").add({
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d"),
                        "name": student_name,
                        "email": email,
                        "state": state_name,
                        "requirement": requirement,
                        "specialization": specialization,
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
                        st.write(f"🎓 **Specialization:** {report.get('specialization', 'N/A')}")
                        st.markdown(
                            report.get("guidance_text", "_No guidance text found._"),
                            unsafe_allow_html=True
                        )
            else:
                st.info("No previous analyses found.")

else:
    st.warning("👋 Please log in or register to access your personalized dashboard.")


