# import streamlit as st
# import fitz  # PyMuPDF
# import spacy
# from spacy.matcher import PhraseMatcher
# import numpy as np
# import joblib
# import tensorflow as tf
# import pandas as pd

# st.set_page_config("PathGenie - Job Fit Analyzer", layout="centered")

# def comma_tokenizer(x):
#     return x.split(',')
# # Load spaCy and ML components
# nlp = spacy.load("en_core_web_sm")
# match_model = tf.keras.models.load_model("match_score_model.h5", compile=False)
# match_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
# match_vectorizer = joblib.load("match_vectorizer.pkl")



# # Load skill vocabulary and job roles
# @st.cache_data
# def load_skill_list():
#     with open("unique_skills.txt", "r") as f:
#         return [line.strip().lower() for line in f if line.strip()]

# @st.cache_data
# def load_role_skills():
#     df = pd.read_csv("job_skills_dataset_corrected.csv")
#     role_map = {}
#     for _, row in df.iterrows():
#         title = row['Job Title'].strip().lower()
#         skills = [s.strip().lower() for s in row['Skills Required'].split(',')]
#         role_map.setdefault(title, set()).update(skills)
#     return role_map

# skill_vocab = load_skill_list()
# role_skill_map = load_role_skills()

# # Skill extractor
# def extract_text_from_pdf(file):
#     doc = fitz.open(stream=file.read(), filetype="pdf")
#     text = "".join([page.get_text() for page in doc])
#     doc.close()
#     return text.lower()

# def extract_skills(text, skill_list):
#     matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
#     patterns = [nlp.make_doc(skill) for skill in skill_list]
#     matcher.add("SKILLS", patterns)
#     doc = nlp(text)
#     matches = matcher(doc)
#     return sorted({doc[start:end].text.lower() for _, start, end in matches})

# # UI

# st.title("📘 Job Fit Analyzer")

# uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
# job_title = st.selectbox("🎯 Target Job Role", sorted(role_skill_map.keys()))
# job_desc = st.text_area("📝 Paste Job Description")
# top_skills = st.multiselect("⭐ Highlight Top Skills", options=sorted(skill_vocab))
# experience = st.number_input("📆 Years of Experience", min_value=0, max_value=50, value=0)

# if st.button("🔍 Analyze Fit"):
#     if not uploaded_file or not job_desc:
#         st.warning("Please upload resume and enter job description.")
#     else:
#         text = extract_text_from_pdf(uploaded_file)
#         resume_skills = extract_skills(text, skill_vocab)
#         jd_skills = extract_skills(job_desc.lower(), skill_vocab)
#         role_skills = role_skill_map.get(job_title.lower(), set())

#         resume_str = ", ".join(resume_skills + top_skills + top_skills)
#         jd_str = ", ".join(jd_skills)
#         role_str = ", ".join(role_skills)

#         vec_resume = match_vectorizer.transform([resume_str]).toarray()[0]
#         vec_jd = match_vectorizer.transform([jd_str]).toarray()[0]
#         vec_role = match_vectorizer.transform([role_str]).toarray()[0]
#         final_input = np.concatenate([vec_resume, vec_jd, vec_role, [experience / 10]])
#         score = match_model.predict(np.array([final_input]))[0][0]

#         # Output
#         st.markdown("---")
#         st.subheader("📊 Results")
#         st.write(f"✅ JD Skill Match: {len(set(resume_skills) & set(jd_skills)) / len(jd_skills) * 100:.1f}%")
#         st.write(f"✅ Role Skill Match: {len(set(resume_skills) & set(role_skills)) / len(role_skills) * 100:.1f}%")
#         st.write(f"🧠 Experience Factor: {experience} years")

#         st.subheader(f"🧮 Overall Profile Fit Score: {score*100:.2f}%")
#         #st.progress(float(score))  # if score is already 0.0–1.0

# # Or if score is 0.0–1.0 and you want to display it as percent:
#         st.progress(float(score) if 0.0 <= score <= 1.0 else float(score) / 100)

import streamlit as st
import fitz  # PyMuPDF
import spacy
from utils import comma_tokenizer
from spacy.matcher import PhraseMatcher
import numpy as np
import joblib
import tensorflow as tf
import pandas as pd
import re


st.set_page_config("PathGenie - Job Fit Analyzer", layout="centered")



# Load spaCy and ML components
nlp = spacy.load("en_core_web_sm")
match_model = tf.keras.models.load_model("match_score_model.h5", compile=False)
match_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
match_vectorizer = joblib.load("match_vectorizer.pkl")

# Load skill vocabulary and job roles
@st.cache_data
def load_skill_list():
    with open("unique_skills.txt", "r") as f:
        return [line.strip().lower() for line in f if line.strip()]

@st.cache_data
def load_role_skills():
    df = pd.read_csv("job_skills_dataset_corrected.csv")
    role_map = {}
    for _, row in df.iterrows():
        title = row['Job Title'].strip().lower()
        skills = [s.strip().lower() for s in row['Skills Required'].split(',')]
        role_map.setdefault(title, set()).update(skills)
    return role_map

skill_vocab = load_skill_list()
role_skill_map = load_role_skills()

# Skill extractor
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = "".join([page.get_text() for page in doc])
    doc.close()
    return text.lower()

def extract_skills(text, skill_list):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in skill_list]
    matcher.add("SKILLS", patterns)
    doc = nlp(text)
    matches = matcher(doc)
    return sorted({doc[start:end].text.lower() for _, start, end in matches})

def extract_experience_from_text(text):
    """
    Extracts years of experience from a job description using regex.
    Returns the first found integer (years), or None if not found.
    """
    # Common patterns: "X years", "X+ years", "at least X years", "minimum X years"
    patterns = [
        r'(\d+)\s*\+\s*years',
        r'at least\s+(\d+)\s*years',
        r'minimum\s+of\s+(\d+)\s*years',
        r'minimum\s+(\d+)\s*years',
        r'(\d+)\s*years'
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None

# UI

st.title("📘 Job Fit Analyzer")

uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
job_title = st.selectbox("🎯 Target Job Role", sorted(role_skill_map.keys()))
job_desc = st.text_area("📝 Paste Job Description")
top_skills = st.multiselect("⭐ Highlight Top Skills", options=sorted(skill_vocab))
experience = st.number_input("📆 Years of Experience", min_value=0, max_value=50, value=0)

if st.button("🔍 Analyze Fit"):
    if not uploaded_file or not job_desc:
        st.warning("Please upload resume and enter job description.")
    else:
        text = extract_text_from_pdf(uploaded_file)
        resume_skills = extract_skills(text, skill_vocab)
        jd_skills = extract_skills(job_desc.lower(), skill_vocab)
        role_skills = role_skill_map.get(job_title.lower(), set())

        resume_str = ", ".join(resume_skills + top_skills + top_skills)
        jd_str = ", ".join(jd_skills)
        role_str = ", ".join(role_skills)

        # Experience matching logic
        required_exp = extract_experience_from_text(job_desc)
        if required_exp is not None:
            # Give more weight if candidate meets or exceeds required experience
            if experience >= required_exp:
                exp_weight = 1.0
                exp_feedback = f"✅ Your experience ({experience} yrs) meets or exceeds the job's requirement ({required_exp} yrs)."
            else:
                exp_weight = experience / required_exp  # 0.0–1.0
                exp_feedback = f"⚠️ Job requires {required_exp} yrs, you have {experience} yrs."
        else:
            exp_weight = experience / 10  # fallback, scale to 0–1
            exp_feedback = "ℹ️ No explicit experience requirement found in job description."

        # Prepare input for model, using weighted experience
        vec_resume = match_vectorizer.transform([resume_str]).toarray()[0]
        vec_jd = match_vectorizer.transform([jd_str]).toarray()[0]
        vec_role = match_vectorizer.transform([role_str]).toarray()[0]
        final_input = np.concatenate([vec_resume, vec_jd, vec_role, [exp_weight]])
        score = match_model.predict(np.array([final_input]))[0][0]

        # Output
        st.markdown("---")
        st.subheader("📊 Results")
        st.write(f"✅ JD Skill Match: {len(set(resume_skills) & set(jd_skills)) / len(jd_skills) * 100:.1f}%" if jd_skills else "No JD skills found.")
        st.write(f"✅ Role Skill Match: {len(set(resume_skills) & set(role_skills)) / len(role_skills) * 100:.1f}%" if role_skills else "No role skills found.")
        st.write(f"🧠 Experience Factor: {experience} years")
        st.info(exp_feedback)
    
        st.subheader(f"🧮 Overall Profile Fit Score: {score*100:.2f}%")
        st.progress(float(score) if 0.0 <= score <= 1.0 else float(score) /100)