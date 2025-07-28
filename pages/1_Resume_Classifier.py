import streamlit as st
import fitz  # PyMuPDF
import joblib
import pandas as pd
import spacy
from utils import comma_tokenizer

from spacy.matcher import PhraseMatcher
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
st.set_page_config("PathGenie - Smart Resume Classifier", layout="centered")
# Load spaCy
nlp = spacy.load("en_core_web_sm")


# Load saved model components
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")
lr_model = joblib.load("logistic_regression_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")
svm_model = joblib.load("svm_model.pkl")

# Load skill vocabulary
def load_skill_list(file="unique_skills.txt"):
    with open(file, "r") as f:
        return [line.strip().lower() for line in f.readlines() if line.strip()]

skill_vocab = load_skill_list()

# Load dataset for calculating role-specific required skills
@st.cache_data
def load_dataset():
    df = pd.read_csv("job_skills_dataset_corrected.csv")
    role_skills = {}
    for _, row in df.iterrows():
        title = row['Job Title'].strip().lower()
        skills = [s.strip().lower() for s in row['Skills Required'].split(',')]
        role_skills.setdefault(title, set()).update(skills)
    return role_skills

role_skill_map = load_dataset()

# Extract text from PDF using fitz
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ''
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.lower()

# Extract skills using spaCy PhraseMatcher
def extract_skills(text, skill_list):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in skill_list]
    matcher.add("SKILLS", patterns)
    doc = nlp(text)
    matches = matcher(doc)
    return sorted({doc[start:end].text.lower() for _, start, end in matches})

# Predict with weighted skills
def predict_with_models(parsed_skills, top_skills):
    final_skills = parsed_skills + top_skills + top_skills  # weight top skills
    text_input = ', '.join(final_skills)
    vec = vectorizer.transform([text_input])
    predictions = {
        "Logistic Regression": label_encoder.inverse_transform(lr_model.predict(vec))[0],
        "Random Forest": label_encoder.inverse_transform(rf_model.predict(vec))[0],
        "SVM": label_encoder.inverse_transform(svm_model.predict(vec))[0],
    }
    return predictions, text_input

# Match % between candidate skills and role skills
def match_score(candidate_skills, job_title):
    required = role_skill_map.get(job_title.lower())
    if not required:
        return "N/A"
    overlap = len(set(candidate_skills) & required)
    return f"{(overlap / len(required)) * 100:.1f}%" if required else "0%"

# UI starts


st.markdown("""
    <style>
    .main-title {
        font-size: 42px;
        color: #4F8BF9;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.3em;
        font-family: 'Segoe UI', sans-serif;
    }
    .subtext {
        text-align: center;
        color: gray;
        font-size: 18px;
        margin-bottom: 2em;
    }
    </style>
    <div class="main-title">PathGenie</div>
    <div class="subtext">Upload your resume, select top skills, and see predicted job roles</div>
""", unsafe_allow_html=True)

# Upload
uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

# Optional top skills
top_skills = st.multiselect("⭐ Highlight Your Top Skills (these get extra weight)", options=sorted(skill_vocab))
all_roles = sorted(set(role_skill_map.keys()))
target_role = st.selectbox("🎯 Your Target Job Role (optional)", options=["None"] + all_roles)
if st.button("🔍 Predict Job Role"):
    if uploaded_file is None:
        st.warning("Please upload a resume.")
    else:
        text = extract_text_from_pdf(uploaded_file)
        parsed_skills = extract_skills(text, skill_vocab)

        if not parsed_skills:
            st.error("❌ No matching skills found in resume.")
        else:
            st.markdown("### ✅ Extracted Skills from Resume")
            st.success(", ".join(parsed_skills))

            st.markdown("### 🧠 Final Prediction with Weighted Top Skills")
            preds, used_text = predict_with_models(parsed_skills, top_skills)

            for model, role in preds.items():
                score = match_score(parsed_skills + top_skills, role)
                st.info(f"🔹 {model}: {role}  🧮 Match Score: {score}")

# If user selected a target role, show learning suggestions
            if target_role != "None":
                    st.markdown("### 📘 Learn & Upskill Recommendations")
                    
                    # Required and current skills
                    required = role_skill_map.get(target_role.lower(), set())
                    current = set(parsed_skills + top_skills)
                    missing = required - current

                    st.markdown("#### 🛠️ Required Skills")
                    st.write(", ".join(sorted(required)) if required else "No data.")

                    st.markdown("#### ✅ You Already Have")
                    st.success(", ".join(sorted(current & required)) if required else "None")

                    st.markdown("#### ❌ You Are Missing")
                    st.warning(", ".join(sorted(missing)) if missing else "You're all set!")

                    # Recommend learning resources for missing skills
                    @st.cache_data
                    def load_resources():
                        return pd.read_csv("learning_resources_dataset.csv")

                    res_df = load_resources()
                    suggestions = res_df[res_df["Skill"].str.lower().isin(missing)]

                    if not suggestions.empty:
                        st.markdown("#### 📗 Suggested Courses")
                        for _, row in suggestions.iterrows():
                            st.markdown(f"- {row['Learning Resource']} — {row['Skill']}")

                    else:
                        st.info("No courses found or you're already skilled enough.")

