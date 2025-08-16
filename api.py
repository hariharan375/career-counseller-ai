import streamlit as st
from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, List
from groq import Groq
import numpy as np
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=api_key)

# State structure
class CounsellorState(TypedDict):
    test_scores: List[Dict[str, int]]  # multiple test scores stored
    state: str
    requirement: str
    student_name: str
    guidance_text: str

# Career Guidance Node
def career_guidance_node(state: CounsellorState):
    # Analyze marks trend
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
    The student {state['student_name']} has the following details:
    - Test Scores: {state['test_scores']}
    - Trends: {trends}
    - Location: {state['state']}
    - Requirement: {state['requirement']}

    Based on these, provide:
    1. Address the student by name and give personalised career guidance with areas needed to improve with the ways to do it.
    2. Mention if their marks trend shows improvement, decline, or stability.
    3. Markdown table of Top 10 suitable colleges for bachelor's degree and the option for masters suggestion also as one markdown table (near their state or nationally reputed).
       - Columns: College Name | Course | Eligibility | Application Process
    4. Give a note to check the respective college website for further details/ clarification
    5. Do NOT return JSON. Format in plain text + valid markdown table.
    """

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
        )
        guidance_output = response.choices[0].message.content.strip()

        return {
            "test_scores": state["test_scores"],
            "state": state["state"],
            "requirement": state["requirement"],
            "student_name": state["student_name"],
            "guidance_text": guidance_output
        }
    except Exception as e:
        return {"guidance_text": f"⚠️ Error: {str(e)}"}

# LangGraph Setup
graph = StateGraph(CounsellorState)
graph.add_node("career_guidance", career_guidance_node)
graph.set_entry_point("career_guidance")
graph.add_edge("career_guidance", END)
app = graph.compile()

# ----------------- STREAMLIT UI -----------------
st.title("🎓 AI Career Counsellor")

# Initialize session state memory
if "test_scores" not in st.session_state:
    st.session_state.test_scores = []

# Student name
student_name = st.text_input("Enter your Name")

# Add test marks
st.subheader("Test Scores")
physics = st.number_input("Physics Marks", min_value=0, max_value=100, key="physics")
maths = st.number_input("Maths Marks", min_value=0, max_value=100, key="maths")
chemistry = st.number_input("Chemistry Marks", min_value=0, max_value=100, key="chemistry")

if st.button("Add Test"):
    st.session_state.test_scores.append({
        "Physics": physics,
        "Maths": maths,
        "Chemistry": chemistry
    })
    st.success("✅ Test added successfully!")

# Display all entered tests
if st.session_state.test_scores:
    st.subheader("📊 Test History")
    for i, test in enumerate(st.session_state.test_scores, 1):
        st.write(f"**Test {i}:** {test}")

# Other inputs
state_name = st.text_input("Enter your State")
requirement = st.text_area("Enter your career interest/requirement")

if st.button("Get Career Guidance"):
    input_state = CounsellorState(
        test_scores=st.session_state.test_scores,
        state=state_name,
        requirement=requirement,
        student_name=student_name,
        guidance_text=""
    )
    final_state = app.invoke(input_state)

    st.subheader("📌 Career Guidance")
    st.markdown(final_state["guidance_text"], unsafe_allow_html=True)

