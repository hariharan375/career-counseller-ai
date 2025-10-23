import streamlit as st
import pandas as pd
import json
import firebase_admin
from firebase_admin import credentials, firestore, auth
from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, List
from groq import Groq

# ============================================================
# ========== FIREBASE INITIALIZATION ==========================
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
        # Local fallback for development
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
    physics_scores = [t["Physics"] for t in state["test_scores"]]
    maths_scores = [t["Maths"] for t in state["test_scores"]]
    chemistry_scores = [t["Chemistry"] for t in state["test_scores"]]

    def get_trend(scores):
        if len(scores) < 2:
            return "Not enough data"
        if scores[-1] > scores[0]:
            return "Improving"
        elif scores[-1] < scores[0]:
            return "Declining"
        else:
            return "Stable"

    trends = {
        "Physics": get_trend(physics_scores),
        "Maths": get_trend(maths_scores),
        "Chemistry": get_trend(chemistry_scores),
    }

    prompt = f"""
    The student named {state['student_name']} has these academic details:
    - Test Scores: {state['test_scores']}
    - Trends: {trends}
    - State: {state['state']}
    - Career Interest: {state['requirement']}

    Provide:
    1. Personalized career guidance with improvement areas and clear action steps.
    2. Mention each subject's trend and what it means.
    3. Markdown table of 10 best **Bachelor’s** colleges related to their interest in India.
    4. Markdown table of 5 best **Master’s** programs (if applicable).
    5. End with a motivational note.
    """

    try:
        # ✅ Updated to a currently supported Groq model
        response = client.chat.completions.create(
            model="llama-3.3-70b-specdec",  # or "mixtral-8x7b"
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
st.set_page_config(page_title="AI Career Counsellor", layout="centered")
st.title("🎓 AI Career Counsellor with Progress Tracking + Groq LLM")
st.caption("An AI-powered system for personalized career guidance and academic analysis.")

# ------------------------------------------------------------
# 🔐 SIDEBAR: Authentication
# ------------------------------------------------------------
st.sidebar.title("🔑 User Authentication")
auth_mode = st.sidebar.radio("Choose Action:", ["Login", "Register"])
email = st.sidebar.text_input("Email")
password = st.sidebar.text_input("Password", type="password")

# Keep user state persistent
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

# ------------------------------------------------------------
# 🧠 MAIN APP SECTION
# ------------------------------------------------------------
user = st.session_state.user
if user:
    st.success(f"Logged in as: {email}")

    # Retrieve test data
    st.session_state.test_scores = []
    tests_ref = db.collection("students").document(user.uid).collection("tests").stream()
    for doc in tests_ref:
        st.session_state.test_scores.append(doc.to_dict())

    # Input new test data
    st.subheader("📚 Enter New Test Marks")
    physics = st.number_input("Physics Marks", 0, 100, 0)
    chemistry = st.number_input("Chemistry Marks", 0, 100, 0)
    maths = st.number_input("Maths Marks", 0, 100, 0)

    if st.button("➕ Add Test"):
        test_data = {"Physics": physics, "Chemistry": chemistry, "Maths": maths}
        db.collection("students").document(user.uid).collection("tests").add(test_data)
        st.session_state.test_scores.append(test_data)
        st.success("✅ Test data added successfully! Please refresh to view updated list.")

    # Display test history
    if st.session_state.test_scores:
        st.subheader("📊 Your Test History")
        df = pd.DataFrame(st.session_state.test_scores)
        st.dataframe(df)

        overall_avg = df.mean().mean()
        st.write(f"**Overall Average Marks:** {overall_avg:.2f}")

        # Additional inputs for AI
        st.subheader("🧠 Get Personalized AI Career Guidance")
        student_name = st.text_input("Your Name")
        state_name = st.text_input("Your State (e.g., Tamil Nadu)")
        requirement = st.text_input("Career Interest (e.g., Engineering, Medicine, Design)")

        if st.button("🚀 Generate Guidance"):
            input_state = CounsellorState(
                student_name=student_name,
                test_scores=st.session_state.test_scores,
                state=state_name,
                requirement=requirement,
                guidance_text=""
            )
            final_state = app.invoke(input_state)

            # Save results
            db.collection("students").document(user.uid).set({
                "name": student_name,
                "email": email,
                "state": state_name,
                "requirement": requirement,
                "last_guidance": final_state["guidance_text"]
            })

            st.subheader("📌 AI Career Guidance")
            st.markdown(final_state["guidance_text"], unsafe_allow_html=True)
    else:
        st.info("No tests added yet. Start by entering your first test data below.")

else:
    st.warning("👋 Please log in or register to access your personalized dashboard.")
