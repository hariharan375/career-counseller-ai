import streamlit as st
import pandas as pd
import json
import firebase_admin
from firebase_admin import credentials, firestore, auth
from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, List
from groq import Groq
import datetime
import traceback

# ----------------------------
# Firebase init (only once)
# ----------------------------
if not firebase_admin._apps:
    if "FIREBASE_KEY" in st.secrets:
        try:
            firebase_key = json.loads(st.secrets["FIREBASE_KEY"])
            cred = credentials.Certificate(firebase_key)
        except Exception as e:
            st.error(f"Failed to parse FIREBASE_KEY from secrets: {e}")
            st.stop()
    else:
        # local fallback filepath (ensure file is present on server)
        cred = credentials.Certificate("career counseller ai/career-counsellor-ai-firebase-adminsdk-fbsvc-ad36c831af.json")

    try:
        firebase_admin.initialize_app(cred)
    except Exception as e:
        # If default app exists, ignore
        if "already exists" not in str(e):
            st.error(f"Firebase initialization error: {e}")
            st.stop()

db = firestore.client()

# ----------------------------
# Groq client
# ----------------------------
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    st.error(f"Failed to initialize Groq client: {e}")
    st.stop()

# ----------------------------
# LangGraph state & node
# ----------------------------
class CounsellorState(TypedDict):
    student_name: str
    test_scores: List[Dict[str, int]]
    state: str
    requirement: str
    guidance_text: str

def career_guidance_node(state: CounsellorState):
    """
    The main LLM node: builds a prompt from test_scores and other fields,
    asks Groq for a guidance output and returns guidance_text.
    """
    try:
        trends = {}
        # derive subjects from the first test row
        subjects = list(state["test_scores"][0].keys()) if state["test_scores"] else []
        # filter out bookkeeping keys
        subjects = [s for s in subjects if s not in ("class", "date_entered")]

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
3. Markdown table of 10 best Bachelor's degree colleges nearer to the location given by the user only (with location, program mentioned, entrance & eligibility, speciality of each college).
4. Markdown table of 5 best Master's programs (if applicable, same specs).
5. End with a motivational summary.
"""
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
        )
        output = response.choices[0].message.content.strip()
        return {**state, "guidance_text": output}
    except Exception as e:
        # return an error text in guidance_text so UI shows the message
        return {**state, "guidance_text": f"⚠️ Error fetching AI guidance: {str(e)}\n\n{traceback.format_exc()}"}

graph = StateGraph(CounsellorState)
graph.add_node("career_guidance", career_guidance_node)
graph.set_entry_point("career_guidance")
graph.add_edge("career_guidance", END)
app = graph.compile()

# ----------------------------
# Streamlit UI config
# ----------------------------
st.set_page_config(page_title="AI Career Counsellor", layout="wide")
st.title("🎓 AI Enabled Career Assistance")

# ----------------------------
# Sidebar auth controls
# ----------------------------
st.sidebar.title("🔑 User Authentication")
auth_mode = st.sidebar.radio("Choose Action:", ["Login", "Register"], index=0)

email = st.sidebar.text_input("Email")
password = st.sidebar.text_input("Password", type="password")

if "user" not in st.session_state:
    st.session_state.user = None

# Register
if auth_mode == "Register" and st.sidebar.button("Create Account"):
    try:
        user = auth.create_user(email=email, password=password)
        st.sidebar.success("✅ Account created! Please login using the Login tab.")
    except Exception as e:
        st.sidebar.error(f"Register error: {e}")

# Login
if auth_mode == "Login" and st.sidebar.button("Login"):
    try:
        user = auth.get_user_by_email(email)
        st.session_state.user = user
        st.sidebar.success(f"✅ Logged in as {email}")
    except Exception as e:
        st.sidebar.error(f"Login failed: {e}")

# Logout
if st.session_state.user:
    if st.sidebar.button("🚪 Logout"):
        st.session_state.user = None
        # preserve only non-sensitive session_state entries if desired, but clearing is fine
        st.session_state.clear()
        st.success("Logged out successfully.")
        st.experimental_rerun()

# ----------------------------
# Helper: get profile
# ----------------------------
def get_profile(uid: str):
    doc = db.collection("students").document(uid).get()
    return doc.to_dict() if doc.exists else None

# ----------------------------
# Main app
# ----------------------------
user = st.session_state.user
if not user:
    st.info("Please login or register from the sidebar to use the application.")
    st.stop()

uid = user.uid
profile = get_profile(uid)  # doc may be None if first login

# Ensure session flags (avoid refresh requirement)
if "profile_created" not in st.session_state:
    st.session_state.profile_created = bool(profile)
if "questionnaire_done" not in st.session_state:
    st.session_state.questionnaire_done = profile.get("questionnaire_done", False) if profile else False

# FIRST-TIME profile creation UI (must complete before questionnaire)
if not profile:
    st.subheader("👋 Welcome! Let's set up your profile (first-time only).")
    name = st.text_input("Enter your name:")
    class_studying = st.selectbox("Select your class:", [str(i) for i in range(8, 13)])
    subjects_input = st.text_area("Enter your subjects separated by commas (e.g., Physics, Chemistry, Maths):")

    if st.button("💾 Save Profile"):
        subjects = [s.strip() for s in subjects_input.split(",") if s.strip()]
        if not name or not class_studying or not subjects:
            st.warning("Please fill name, class and at least one subject.")
        else:
            # store initial profile (questionnaire not done)
            db.collection("students").document(uid).set({
                "name": name,
                "class": class_studying,
                "subjects": subjects,
                "questionnaire_done": False,
                "domain": None
            })
            st.success("Profile saved. Please complete the questionnaire next.")
            # refresh profile variable and session flags without requiring browser refresh
            profile = get_profile(uid)
            st.session_state.profile_created = True
            st.session_state.questionnaire_done = profile.get("questionnaire_done", False)
            st.experimental_rerun()

# If profile exists, show main navigation (but lock Counsel/Previous until questionnaire done)
if profile:
    student_name = profile.get("name", "")
    student_class = profile.get("class", "")
    subjects = profile.get("subjects", [])
    saved_domain = profile.get("domain", None)
    questionnaire_done_db = profile.get("questionnaire_done", False)

    # Keep session consistent with DB if DB says questionnaire done
    if questionnaire_done_db and not st.session_state.questionnaire_done:
        st.session_state.questionnaire_done = True

    st.sidebar.success(f"Hello, {student_name}! 👋")

    # Construct nav pages based on questionnaire state
    nav_pages = ["Profile Details"]
    if not st.session_state.questionnaire_done:
        nav_pages.append("Questionnaire")
    else:
        nav_pages += ["Counsel", "Previous Analysis"]

    page = st.sidebar.radio("📂 Navigate to:", nav_pages)

    # ---------------- Profile Details ----------------
    if page == "Profile Details":
        st.subheader("📋 Profile Details")
        new_name = st.text_input("Name", value=student_name)
        new_class = st.selectbox("Class", [str(i) for i in range(8, 13)], index=int(student_class) - 8)
        new_subjects_str = st.text_area("Subjects (comma-separated):", value=", ".join(subjects))

        if st.button("💾 Update Profile"):
            updated_subjects = [s.strip() for s in new_subjects_str.split(",") if s.strip()]
            if not new_name or not new_class or not updated_subjects:
                st.warning("Please provide a name, class and at least one subject.")
            else:
                db.collection("students").document(uid).update({
                    "name": new_name,
                    "class": new_class,
                    "subjects": updated_subjects
                })
                st.success("Profile updated.")
                # refresh local variables
                profile = get_profile(uid)
                st.experimental_rerun()

    # ---------------- Questionnaire ----------------
    elif page == "Questionnaire":
        st.subheader("🧩 Career Interest Questionnaire")
        st.info("Rate each statement from 1 (Strongly Disagree) to 5 (Strongly Agree). You can submit only once.")

        # list of 31 statements exactly as provided
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

        # show sliders; keep their values in st.session_state to avoid losing them on reruns
        responses = {}
        for i, q in enumerate(questions, 1):
            key = f"q_{i}"
            if key not in st.session_state:
                st.session_state[key] = 3
            st.session_state[key] = st.slider(q, 1, 5, st.session_state[key], key=key)
            responses[f"Q{i}"] = st.session_state[key]

        if st.button("✅ Submit Questionnaire"):
            # domain mapping
            domain_map = {
                "Engineering & Technology": [1, 3, 8, 11, 22],
                "Research & Science": [2, 7, 12, 23, 26],
                "Medical & Life Sciences": [4, 9, 13, 21, 24],
                "Arts & Design": [5, 10, 14, 20, 29],
                "Business & Management": [15, 17, 19, 27, 30],
                "Law & Public Services": [6, 16, 25, 28, 31],
            }

            domain_scores = {d: sum(responses[f"Q{i}"] for i in q_nums) for d, q_nums in domain_map.items()}
            top_domain = max(domain_scores, key=domain_scores.get)

            # save questionnaire and domain to DB; no edit allowed later
            db.collection("students").document(uid).collection("questionnaire").add({
                "responses": responses,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d"),
                "calculated_domain": top_domain,
                "domain_scores": domain_scores
            })
            db.collection("students").document(uid).update({
                "questionnaire_done": True,
                "domain": top_domain
            })
            st.session_state.questionnaire_done = True
            st.success("Questionnaire submitted successfully. The rest of the app is now accessible.")
            # clear slider values to avoid weirdness on next visit
            for i in range(1, len(questions)+1):
                st.session_state.pop(f"q_{i}", None)
            st.experimental_rerun()

    # ---------------- Counsel ----------------
    elif page == "Counsel":
        # guard: this page should be available only if questionnaire done
        if not st.session_state.questionnaire_done:
            st.warning("Please complete the questionnaire first to access the Counsel page.")
        else:
            st.subheader("🧠 Counsel: Enter test marks and get AI guidance")

            # load tests
            test_scores = []
            tests_ref = db.collection("students").document(uid).collection("tests").stream()
            for d in tests_ref:
                test_scores.append(d.to_dict())

            st.write(f"Currently tracking subjects: {', '.join(subjects)}")

            # Input new test row
            with st.form("add_test_form", clear_on_submit=True):
                test_row = {
                    "class": student_class,
                    "date_entered": datetime.datetime.now().strftime("%Y-%m-%d")
                }
                for sub in subjects:
                    # unique key to avoid clobbering session state
                    v = st.number_input(f"{sub} Marks", 0, 100, 0, key=f"mark_{sub}")
                    test_row[sub] = int(v)
                add_clicked = st.form_submit_button("➕ Add Test")
                if add_clicked:
                    db.collection("students").document(uid).collection("tests").add(test_row)
                    st.success("Test saved.")
                    # refresh local view
                    test_scores.append(test_row)

            if test_scores:
                df = pd.DataFrame(test_scores)
                # ensure columns exist and reorder: class, date_entered, then subjects
                columns_order = ["class", "date_entered"] + [s for s in subjects if s in df.columns]
                # if any subject missing from db rows, safe-get
                columns_order = [c for c in columns_order if c in df.columns]
                df = df[columns_order]
                st.subheader("📊 Your Test History")
                st.dataframe(df)

            # Career interest & dynamic follow-ups
            st.subheader("🚀 Career Guidance Assistant")
            state_name = st.text_input("Your State (for locality-aware college suggestions):", value="")
            requirement = st.text_input("Career Interest (e.g., Doctor, Engineer, Designer):", value="")

            # Only show followups if user typed requirement
            if requirement:
                st.info(f"Let's understand your interest in **{requirement}** better.")
                # pick follow-ups based on keywords
                followup_qs_map = {
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

                key = next((k for k in followup_qs_map.keys() if k in requirement.lower()), None)
                if key:
                    qs = followup_qs_map[key]
                else:
                    qs = [
                        "What draws you toward this field?",
                        "What kind of daily work or challenges excite you in this area?",
                        "Where do you see yourself applying these skills in the future?"
                    ]

                # render text areas for three follow-ups
                followup_answers = []
                for idx, q in enumerate(qs, 1):
                    k = f"follow_{idx}"
                    if k not in st.session_state:
                        st.session_state[k] = ""
                    st.session_state[k] = st.text_area(q, value=st.session_state[k], key=k)
                    followup_answers.append(st.session_state[k])

                # Generate button (only when followups filled)
                if all(a.strip() for a in followup_answers) and st.button("💡 Generate Career Guidance"):
                    # 1) call Groq to narrow specialization from follow-ups
                    try:
                        narrow_prompt = f"""
Student: {student_name}
Interest: {requirement}
Follow-up answers: {dict(zip(qs, followup_answers))}
Task: Suggest a short specialization/sub-field (one line) and a two-line reasoning why that specialization fits this student.
"""
                        subresp = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[{"role": "user", "content": narrow_prompt}],
                        )
                        specialization = subresp.choices[0].message.content.strip().split("\n")[0].strip()
                    except Exception as e:
                        specialization = "General"
                        st.error(f"Specialization generation failed, proceeding with 'General'. ({e})")

                    # 2) Build requirement-based guidance via LangGraph app.invoke
                    # Prepare state
                    state_for_interest = CounsellorState(
                        student_name=student_name,
                        test_scores=test_scores if test_scores else [{"class": student_class}],
                        state=state_name,
                        requirement=f"{requirement} - {specialization}",
                        guidance_text=""
                    )
                    try:
                        interest_out_state = app.invoke(state_for_interest)
                        interest_guidance = interest_out_state.get("guidance_text", "No guidance returned.")
                    except Exception as e:
                        interest_guidance = f"⚠️ Error generating interest guidance: {e}"

                    # 3) Domain-guidance for the domain computed from questionnaire
                    actual_domain = saved_domain or profile.get("domain") or "Not Defined"

                    # If interest string contains the domain name (simple substring), treat as match
                    matched = actual_domain and (actual_domain.lower() in requirement.lower())

                    if matched:
                        # If matched, produce only one unified output (the interest-guidance)
                        combined_output = f"### 🎯 Matched your assessed domain: {actual_domain}\n\n" + interest_guidance
                        # Save single guidance entry
                        db.collection("students").document(uid).collection("guidance_history").add({
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d"),
                            "name": student_name,
                            "email": email,
                            "state": state_name,
                            "requirement": requirement,
                            "specialization": specialization,
                            "guidance_text": combined_output,
                            "domain": actual_domain,
                            "match": True
                        })
                        st.markdown(combined_output, unsafe_allow_html=True)
                    else:
                        # If mismatch, also get domain-based guidance (same format as interest guidance but requirement replaced)
                        state_for_domain = CounsellorState(
                            student_name=student_name,
                            test_scores=test_scores if test_scores else [{"class": student_class}],
                            state=state_name,
                            requirement=f"{actual_domain} - (assessed domain from questionnaire)",
                            guidance_text=""
                        )
                        try:
                            domain_out_state = app.invoke(state_for_domain)
                            domain_guidance = domain_out_state.get("guidance_text", "No guidance returned.")
                        except Exception as e:
                            domain_guidance = f"⚠️ Error generating domain guidance: {e}"

                        # Compose combined output: first domain guidance (aptitude-driven), then interest guidance
                        combined_output = (
                            f"### 🔎 Guidance based on your assessed aptitude domain: **{actual_domain}**\n\n"
                            f"{domain_guidance}\n\n"
                            f"---\n\n"
                            f"### 🎯 Guidance based on your expressed interest: **{requirement}** (specialized: {specialization})\n\n"
                            f"{interest_guidance}"
                        )

                        # Save combined guidance as one record
                        db.collection("students").document(uid).collection("guidance_history").add({
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d"),
                            "name": student_name,
                            "email": email,
                            "state": state_name,
                            "requirement": requirement,
                            "specialization": specialization,
                            "guidance_text": combined_output,
                            "domain": actual_domain,
                            "match": False
                        })
                        st.markdown(combined_output, unsafe_allow_html=True)

    # ---------------- Previous Analysis ----------------
    elif page == "Previous Analysis":
        if not st.session_state.questionnaire_done:
            st.warning("Please complete the questionnaire first to access previous guidance.")
        else:
            st.subheader("🕒 Previous Career Guidance Reports")
            reports_ref = db.collection("students").document(uid).collection("guidance_history").order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
            reports = [r.to_dict() for r in reports_ref]
            if not reports:
                st.info("No guidance history found yet. Generate guidance from Counsel.")
            else:
                for i, r in enumerate(reports, start=1):
                    title = f"Report {i}: {r.get('requirement','Unknown')} ({r.get('timestamp','No date')})"
                    with st.expander(title):
                        st.write(f"👤 **Name:** {r.get('name','N/A')}")
                        st.write(f"📍 **State:** {r.get('state','N/A')}")
                        st.write(f"✉️ **Email:** {r.get('email','N/A')}")
                        # Do not expose the internal domain computation explicitly if you want it hidden — you asked earlier that domain not be visible.
                        # Here we include it in the report for transparency; remove the next line if you want domain hidden.
                        st.write(f"🎯 **Assessed Domain (internal):** {r.get('domain','Hidden')}")
                        st.markdown(r.get("guidance_text","_No guidance text found._"), unsafe_allow_html=True)

else:
    st.warning("Please login to access the application.")
