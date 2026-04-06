import streamlit as st
import PyPDF2
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
# ---------------- CONFIG ----------------
st.set_page_config(page_title="Placement Readiness System", layout="wide")

# ---------------- LEARNING RESOURCES ----------------
learning_resources = {

    "coding": {
        "youtube": [
            ("NeetCode DSA Playlist", "https://www.youtube.com/c/NeetCode"),
            ("Abdul Bari Data Structures", "https://www.youtube.com/playlist?list=PLBlnK6fEyqRgLLlzdgiTUKULKJPYc0A4q")
        ],
        "pdf": "resources/python basic programes.pdf"
    },

    "dsa": {
        "youtube": [
            ("Striver DSA Sheet", "https://www.youtube.com/c/takeUforward"),
            ("Love Babbar DSA Playlist", "https://www.youtube.com/c/LoveBabbar1")
        ],
        "pdf": "resources/DSA complete Cheatsheet.pdf"
    },

    "communication": {
        "youtube": [
            ("HR Interview Preparation", "https://www.youtube.com/watch?v=HG68Ymazo18"),
            ("Communication Skills Tips", "https://www.youtube.com/watch?v=HAnw168huqA")
        ],
        "pdf": "resources/communication_notes.pdf"
    }
}

# ---------------- PROJECT IDEAS ----------------
project_ideas = {

    "machine_learning": [
        "Fake News Detection using NLP",
        "Movie Recommendation System",
        "Spam Email Classifier",
        "Student Performance Predictor"
    ],

    "python": [
        "Password Strength Checker",
        "File Organizer Tool",
        "CLI To-Do List Application",
        "Python Quiz Application"
    ],

    "web_apps": [
        "AI Chatbot using Streamlit",
        "Resume Analyzer using NLP",
        "Portfolio Website using Flask"
    ]
}

# ---------------- LOAD MODEL ----------------
model = joblib.load("placement_lr_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("placement_data.csv")
X = df.drop("placed", axis=1)
y = df["placed"]
X_scaled = scaler.transform(X)

# ---------------- HEADER ----------------
st.title("🎯 Placement Readiness & Skill Development System")

st.write(
"This system predicts placement readiness using Machine Learning "
"and provides personalized learning recommendations."
)

st.divider()

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("🧑‍🎓 Student Profile")

cgpa = st.sidebar.slider("CGPA", 5.0, 10.0, 7.5)
coding = st.sidebar.slider("Coding Skill (0-10)", 0, 10, 5)
dsa = st.sidebar.slider("DSA Practice Level (0-10)", 0, 10, 5)
projects = st.sidebar.number_input("Projects Completed", 0, 10, 2)
internship = st.sidebar.selectbox("Internship Experience", ["No", "Yes"])
communication = st.sidebar.slider("Communication Skill (0-10)", 0, 10, 5)
certifications = st.sidebar.number_input("Certifications", 0, 10, 1)
mock_score = st.sidebar.slider("Mock Interview Score", 0, 100, 60)

internship_val = 1 if internship == "Yes" else 0

input_data = np.array([[cgpa, coding, dsa, projects,
                        internship_val, communication,
                        certifications, mock_score]])

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs([
    "📊 Prediction Analysis",
    "🛠 Skill Development Roadmap",
    "📄 Resume Analyzer"
])

# =====================================================
# TAB 1 : PREDICTION
# =====================================================
with tab1:

    st.subheader("Placement Probability Analysis")

    if st.button("Predict Readiness"):

        scaled_input = scaler.transform(input_data)

        probability = model.predict_proba(scaled_input)[0][1] * 100
        y_prob_all = model.predict_proba(X_scaled)[:,1]

        col1, col2 = st.columns([1,2])

        with col1:

            st.metric("Placement Probability", f"{probability:.2f}%")

            if probability >= 75:
                st.success("Status : Placement Ready")

            elif probability >= 50:
                st.warning("Status : Almost Ready")

            else:
                st.error("Status : Needs Improvement")

        with col2:

            fpr, tpr, _ = roc_curve(y, y_prob_all)
            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots(figsize=(4,3))
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0,1],[0,1],"--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()

            st.pyplot(fig)

        st.divider()

        st.subheader("Feature Impact Analysis")

        coef_df = pd.DataFrame({
            "Feature":X.columns,
            "Impact":model.coef_[0]
        }).sort_values(by="Impact",ascending=False)

        st.dataframe(coef_df,use_container_width=True)

# =====================================================
# TAB 2 : ROADMAP
# =====================================================
with tab2:

    st.subheader("Personalized Skill Improvement Plan")

    st.write("Based on your profile the following areas can be improved:")

    st.divider()

    # ---------- CODING ----------
    if coding < 6:

        with st.expander("💻 Improve Coding Skills"):

            st.markdown("### 📺 Video Resources")

            for title,link in learning_resources["coding"]["youtube"]:
                st.markdown(f"- [{title}]({link})")

            st.markdown("### 📄 Download Notes")

            with open(learning_resources["coding"]["pdf"],"rb") as file:
                st.download_button(
                    "Download Coding Notes",
                    data=file,
                    file_name="python_basic_programs.pdf"
                )

            st.markdown("### 📝 Practice Plan")
            st.write("- Solve 5 coding problems daily")
            st.write("- Focus on Arrays, Strings and HashMaps")

    # ---------- DSA ----------
    if dsa < 6:

        with st.expander("🧠 Improve Data Structures & Algorithms"):

            st.markdown("### 📺 Video Resources")

            for title,link in learning_resources["dsa"]["youtube"]:
                st.markdown(f"- [{title}]({link})")

            st.markdown("### 📄 Download Notes")

            with open(learning_resources["dsa"]["pdf"],"rb") as file:
                st.download_button(
                    "Download DSA Notes",
                    data=file,
                    file_name="dsa_cheatsheet.pdf"
                )

            st.markdown("### 📝 Practice Plan")
            st.write("- Study Arrays → Linked Lists → Trees → Graphs")

    # ---------- PROJECT IDEAS ----------
    if projects < 3:

        with st.expander("📂 Build Strong Projects"):

            st.markdown("### Suggested Projects")

            if coding >= 6 and dsa >= 6:

                for idea in project_ideas["machine_learning"]:
                    st.write("-",idea)

            elif coding >= 5:

                for idea in project_ideas["python"]:
                    st.write("-",idea)

            else:

                for idea in project_ideas["web_apps"]:
                    st.write("-",idea)

            st.write("Tip: Upload projects to GitHub with proper documentation")

    # ---------- INTERNSHIP ----------
    if internship_val == 0:

        with st.expander("🏢 Gain Internship Experience"):

            st.write("- Apply on LinkedIn")
            st.write("- Apply through Internshala")
            st.write("- Contribute to open source")

    # ---------- COMMUNICATION ----------
    if communication < 6:

        with st.expander("🗣 Improve Communication Skills"):

            st.markdown("### 📺 Video Resources")

            for title,link in learning_resources["communication"]["youtube"]:
                st.markdown(f"- [{title}]({link})")

            st.markdown("### 📄 Download Notes")

            with open(learning_resources["communication"]["pdf"],"rb") as file:
                st.download_button(
                    "Download Communication Notes",
                    data=file,
                    file_name="communication_notes.pdf"
                )

            st.markdown("### 📝 Practice Plan")
            st.write("- Practice mock interviews")
            st.write("- Improve confidence in speaking")

    if coding>=6 and dsa>=6 and projects>=3 and communication>=6:

        st.success("🎉 You are on the right track!")

    st.divider()

    st.info("This system is a decision-support tool and does not replace actual recruitment processes.")

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text.lower()


with tab3:
    role = st.selectbox(
    "🎯 Select Target Role",
    ["Software Developer (SDE)", "Data Scientist"]
)
    st.subheader("📄 Resume Analyzer & ATS Score")

    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

    if uploaded_file is not None:

        resume_text = extract_text_from_pdf(uploaded_file)

        st.success("Resume Uploaded Successfully")

        # ---------------- SKILLS ----------------
        required_skills = [
            "python", "java", "c++",
            "data structures", "algorithms",
            "machine learning", "sql",
            "projects", "internship",
            "communication"
        ]

        found_skills = []
        missing_skills = []

        for skill in required_skills:
            if skill in resume_text:
                found_skills.append(skill)
            else:
                missing_skills.append(skill)

        # ---------------- ATS SCORE ----------------
        ats_score = int((len(found_skills) / len(required_skills)) * 100)

        col1, col2 = st.columns([1, 2])

        with col1:

            st.metric("🎯 ATS Score", f"{ats_score}%")

            # Color-based status
            if ats_score >= 75:
                st.success("🟢 High Chance of Selection")
            elif ats_score >= 50:
                st.warning("🟡 Moderate Chance")
            else:
                st.error("🔴 Low Chance - Improve Resume")

            # Progress bar
            st.progress(ats_score / 100)

        with col2:

            # Donut Chart
            labels = ["Matched Skills", "Missing Skills"]
            sizes = [len(found_skills), len(missing_skills)]

            fig, ax = plt.subplots()
            ax.pie(
                sizes,
                labels=labels,
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops=dict(width=0.4)
            )
            ax.axis("equal")

            st.pyplot(fig)

        st.divider()

        # ---------------- SKILLS DISPLAY ----------------
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("✅ Skills Found")
            if found_skills:
                for skill in found_skills:
                    st.success(skill)
            else:
                st.warning("No matching skills found")

        with col4:
            st.subheader("❌ Missing Skills")
            for skill in missing_skills:
                st.error(skill)

        st.divider()

        # ---------------- RECOMMENDATIONS ----------------
        st.subheader("📌 Smart Recommendations")

        if missing_skills:

            for skill in missing_skills:

                if skill in ["data structures", "algorithms"]:
                    st.write("➡ Improve DSA using Striver or NeetCode")

                elif skill == "machine learning":
                    st.write("➡ Add ML project (e.g., prediction system)")

                elif skill == "projects":
                    st.write("➡ Add at least 2 strong projects with GitHub links")

                elif skill == "internship":
                    st.write("➡ Gain internship experience via Internshala or LinkedIn")

                elif skill == "communication":
                    st.write("➡ Practice HR interview communication")

                else:
                    st.write(f"➡ Learn and add {skill} to your resume")

        else:
            st.success("🎉 Excellent Resume! All key skills present.")