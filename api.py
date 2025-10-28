import streamlit as st
import pandas as pd
import json
import firebase_admin
from firebase_admin import credentials, firestore, auth
from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, List
from groq import Groq
import datetime
import uuid

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
        cred = credentials.Certificate(
            "career counseller ai/career-counsellor-ai-firebase-adminsdk-fbsvc-ad36c831af.json"
        )
    firebase_admin.initialize_app(cred)

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
    physics_scores = [t.get("Physics", 0) for t in state["test_scores"]]
    maths_scores = [t.get("Maths", 0) for t in state["test_scores"]]
    chemistry_scores = [t.get("Chemistry", 0) for t in state["test_scores"]]

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
    3. Markdown table of 10 best Bachelor's colleges related to their interest according to the marks and capability nearby their given location, preferably with eligibility criteria and NIRF ranking 2025 in separate columns.
    4. Markdown table of 5 best Master's programs (if applicable).
    5. End with a brief summary about feasible career options and their capabilities for the same, followed by a motivational note.
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
# ========== HELPERS FOR VERSIONS & PROFILE ==================
# ============================================================
def get_profile(uid):
    doc = db.collection("students").document(uid).get()
    if doc.exists:
        return doc.to_dict()
    return None

def save_profile(uid, name, current_class):
    db.collection("students").document(uid).set({
        "name": name,
        "class": current_class,
        "updated_at": datetime.datetime.now().strftime("%Y-%m-%d")
    }, merge=True)

def create_version(uid, subjects_list, label=None):
    """Create a new version document for subject set. Returns version_id."""
    version_id = str(uuid.uuid4())
    doc = {
        "subjects": subjects_list,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d"),
        "label": label or f"v_{datetime.datetime.now().strftime('%Y%m%d')}"
    }
    db.collection("students").document(uid).collection("versions").document(version_id).set(doc)
    return version_id

def list_versions(uid):
    versions = db.collection("students").document(uid).collection("versions").stream()
    result = []
    for v in versions:
        data = v.to_dict()
        data["_id"] = v.id
        result.append(data)
    # sort by created_at descending
    result.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return result

def get_version(uid, version_id):
    doc = db.collection("students").document(uid).collection("versions").document(version_id).get()
    if doc.exists:
        d = doc.to_dict()
        d["_id"] = doc.id
        return d
    return None

def add_test_to_version(uid, version_id, test_data):
    """
    test_data: dict of subject->score, plus 'Class' and 'Date'
    """
    db.collection("students").document(uid).collection("versions").document(version_id).collection("tests").add(test_data)

def list_tests_for_version(uid, version_id):
    docs = db.collection("students").document(uid).collection("versions").document(version_id).collection("tests").stream()
    rows = []
    for d in docs:
        r = d.to_dict()
        rows.append(r)
    # if Date present sort descending
    try:
        rows.sort(key=lambda x: x.get("Date", ""), reverse=True)
    except:
        pass
    return rows

# ============================================================
# ========== STREAMLIT UI & AUTH ==============================
# ============================================================
st.set_page_config(page_title="AI Career Counsellor", layout="wide")
st.title("🎓 AI Enabled Career Assistance")
st.caption("An AI-powered system for personalized career guidance and academic analysis.")

# --- Sidebar: Authentication & navigation ---
st.sidebar.title("🔑 Authentication")
auth_mode = st.sidebar.radio("Action", ["Login", "Register"])
email = st.sidebar.text_input("Email")
password = st.sidebar.text_input("Password", type="password")

if "user" not in st.session_state:
    st.session_state.user = None

if auth_mode == "Register" and st.sidebar.button("Create Account"):
    try:
        user_obj = auth.create_user(email=email, password=password)
        st.sidebar.success("Account created. Please login.")
    except Exception as e:
        st.sidebar.error(f"Register error: {e}")

if auth_mode == "Login" and st.sidebar.button("Login"):
    try:
        user_obj = auth.get_user_by_email(email)
        st.session_state.user = user_obj
        st.sidebar.success(f"Logged in as {email}")
    except Exception as e:
        st.sidebar.error(f"Login error: {e}")

# Logout button
if st.session_state.user:
    if st.sidebar.button("🚪 Logout"):
        # clear only keys we set
        for k in ["user", "current_version", "profile_loaded"]:
            if k in st.session_state:
                del st.session_state[k]
        st.success("Logged out.")
        st.experimental_rerun()

# If not logged in, show message and stop.
if not st.session_state.user:
    st.info("Please login or register from the sidebar to continue.")
    st.stop()

# =====> From here the user is logged in:
user = st.session_state.user
uid = user.uid

# Load or create profile
profile = get_profile(uid)
if profile is None or not profile.get("name") or not profile.get("class"):
    st.subheader("Welcome — set up your profile")
    # ask for name & class (only once)
    name_input = st.text_input("Full name")
    class_input = st.selectbox("Class / Grade", options=[str(i) for i in range(8, 13)])
    if st.button("Save Profile"):
        if not name_input:
            st.error("Please enter your name.")
        else:
            save_profile(uid, name_input, class_input)
            st.success("Profile saved. Refreshing...")
            st.experimental_rerun()
else:
    # Greet user by name
    st.sidebar.success(f"Hello {profile.get('name')}")

# reload profile after possible creation
profile = get_profile(uid)
student_name = profile.get("name")
student_class_profile = profile.get("class")

# === Subjects / versions management ===
st.sidebar.subheader("Subjects & Versions")

versions = list_versions(uid)
version_labels = [f"{v.get('label')} — {v.get('created_at')}" for v in versions]
version_ids = [v["_id"] for v in versions]
# Determine current/latest version or let user pick
if "current_version" not in st.session_state:
    if versions:
        st.session_state.current_version = versions[0]["_id"]
    else:
        st.session_state.current_version = None

# Show current subjects if exists
if st.session_state.current_version:
    current_ver = get_version(uid, st.session_state.current_version)
    st.sidebar.markdown("**Current Subjects (latest version):**")
    st.sidebar.write(", ".join(current_ver.get("subjects", [])))
else:
    st.sidebar.info("No subjects defined. Create a set below.")

# Option to select which version to work on (for adding tests / viewing)
if versions:
    sel_idx = st.sidebar.selectbox("Select version to work with", options=list(range(len(versions))), format_func=lambda i: version_labels[i])
    sel_version_id = version_ids[sel_idx]
    st.session_state.current_version = sel_version_id
else:
    sel_version_id = None

st.sidebar.markdown("---")
st.sidebar.subheader("Create / Edit Subjects")
# provide a list of common subjects and allow custom addition
common_subjects = ["Physics", "Chemistry", "Maths", "Biology", "English", "Computer Science", "History", "Civics", "Economics"]
selected = st.sidebar.multiselect("Choose subjects (check)", options=common_subjects)
custom = st.sidebar.text_input("Any other subjects (comma separated)?")
if custom:
    custom_list = [s.strip() for s in custom.split(",") if s.strip()]
else:
    custom_list = []
new_subjects = list(dict.fromkeys(selected + custom_list))  # preserve order & dedupe

if st.sidebar.button("Save as new subject set (creates new version)"):
    if not new_subjects:
        st.sidebar.error("Please choose or enter at least one subject.")
    else:
        new_vid = create_version(uid, new_subjects, label=f"Subjects_{len(versions)+1}")
        st.sidebar.success(f"Created version {new_vid}")
        # update session state to new version
        st.session_state.current_version = new_vid
        st.experimental_rerun()

# Option to set selected version as the active one for convenience
if st.sidebar.button("Use selected version as active"):
    if sel_version_id:
        st.session_state.current_version = sel_version_id
        st.sidebar.success("Active version updated.")
    else:
        st.sidebar.error("No version available.")

st.sidebar.markdown("---")
st.sidebar.subheader("Navigation")
page = st.sidebar.radio("Go to:", ["Dashboard", "Previous Analysis"])

# -------------------- MAIN PAGES --------------------
if page == "Dashboard":
    st.header("📊 Dashboard")

    # Show profile summary and allow editing
    with st.expander("Profile (click to edit)"):
        st.write(f"**Name:** {student_name}")
        st.write(f"**Class (profile):** {student_class_profile}")
        if st.button("Edit profile"):
            # simple inline edit
            new_name = st.text_input("New full name", value=student_name)
            new_class = st.selectbox("New class", options=[str(i) for i in range(8, 13)], index=int(student_class_profile)-8 if student_class_profile and student_class_profile.isdigit() else 0)
            if st.button("Save changes"):
                if not new_name:
                    st.error("Name required")
                else:
                    save_profile(uid, new_name, new_class)
                    st.success("Profile updated")
                    st.experimental_rerun()

    # Which version are we adding tests to?
    if st.session_state.current_version is None:
        st.warning("No subject set available. Please create a subject set in sidebar to start adding tests.")
        st.stop()

    current_version_doc = get_version(uid, st.session_state.current_version)
    subjects = current_version_doc.get("subjects", [])

    st.subheader(f"Entering tests for subjects set: {current_version_doc.get('label')} ({current_version_doc.get('created_at')})")
    st.write("Subjects:", ", ".join(subjects))

    # Allow changing class per test entry optionally (prefilled with profile class)
    test_class = st.selectbox("Class for this test (you can change):", options=[str(i) for i in range(8, 13)], index=int(student_class_profile)-8 if student_class_profile and student_class_profile.isdigit() else 0)

    # dynamic inputs for subjects
    st.markdown("Enter marks for each subject:")
    marks = {}
    for sub in subjects:
        marks[sub] = st.number_input(f"{sub} Marks", min_value=0, max_value=100, value=0, key=f"mark_{sub}")

    if st.button("➕ Add Test"):
        # store with date only
        test_date = datetime.datetime.now().strftime("%Y-%m-%d")
        test_data = {"Class": test_class, "Date": test_date}
        test_data.update(marks)
        add_test_to_version(uid, st.session_state.current_version, test_data)
        st.success("Test added to selected version.")
        # reload page to show updated tests
        st.experimental_rerun()

    # Show test history grouped by versions (each version a table)
    st.subheader("Your Test Histories (by subject-set version)")

    versions_all = list_versions(uid)  # fresh list
    if not versions_all:
        st.info("No versions / subject-sets found.")
    else:
        for v in versions_all:
            st.markdown(f"### Version: {v.get('label')} — created on {v.get('created_at')}")
            tests = list_tests_for_version(uid, v["_id"])
            if tests:
                df = pd.DataFrame(tests)
                # Remove internal Date if you don't want to display time -> we saved only Date. Keep Date visible (you previously asked to remove timestamp; here it is Date)
                # If you want to hide Date column comment out next two lines:
                # if "Date" in df.columns:
                #     df = df.drop(columns=["Date"])
                st.dataframe(df)
            else:
                st.info("No tests in this version yet.")

    # AI Guidance section (uses profile name)
    st.subheader("🧠 AI Career Guidance")
    st.write(f"Guidance will address: **{student_name}**")
    state_name = st.text_input("Your State (for college locality)", value=profile.get("state", ""))
    requirement = st.text_input("Career Interest (e.g., Engineering, Medicine, Design)")
    if st.button("🚀 Generate Guidance"):
        # Build flattened tests across all versions for the LLM input or use current version tests — choose current_version tests
        all_tests_flat = []
        # Option: pass tests from all versions or only current version; using combined tests here
        for v in list_versions(uid):
            tests_v = list_tests_for_version(uid, v["_id"])
            # include version label for context
            for t in tests_v:
                t_copy = dict(t)
                t_copy["_version_label"] = v.get("label")
                all_tests_flat.append(t_copy)

        input_state = CounsellorState(
            student_name=student_name,
            test_scores=all_tests_flat,
            state=state_name,
            requirement=requirement,
            guidance_text=""
        )
        final_state = app.invoke(input_state)

        # Save guidance to guidance_history with date
        db.collection("students").document(uid).collection("guidance_history").add({
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d"),
            "name": student_name,
            "email": email,
            "state": state_name,
            "requirement": requirement,
            "guidance_text": final_state["guidance_text"]
        })

        st.subheader("📌 AI Career Guidance")
        st.markdown(final_state["guidance_text"], unsafe_allow_html=True)

elif page == "Previous Analysis":
    st.header("🕒 Previous Analysis")
    st.write(f"Showing previous AI guidance for **{get_profile(uid).get('name', '')}**")

    reports_ref = (
        db.collection("students")
        .document(uid)
        .collection("guidance_history")
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .stream()
    )
    reports = [r.to_dict() for r in reports_ref]

    if not reports:
        st.info("No previous guidance history found.")
    else:
        for idx, r in enumerate(reports, start=1):
            with st.expander(f"{r.get('timestamp','No date')} — {r.get('requirement','No interest')}"):
                st.write(f"**Name:** {r.get('name')}")
                st.write(f"**State:** {r.get('state')}")
                st.markdown("---")
                st.markdown(r.get("guidance_text", "_No guidance text_"), unsafe_allow_html=True)
