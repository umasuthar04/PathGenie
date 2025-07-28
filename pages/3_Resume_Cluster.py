# def custom_tokenizer(text):
#     return [token.strip() for token in text.split(',') if token.strip()]
from utils import comma_tokenizer
import streamlit as st
import tempfile
from resume_parser import parse_pdf_for_skills
from cluster_predictor import predict_resume_clusters

st.title("Resume Skill Cluster Predictor")

st.write(
    "Upload your resume PDF. The app will extract your skills and predict the top 3 matching job clusters."
)

uploaded_file = st.file_uploader("Choose your resume PDF", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    # Extract skills and show them
    with st.spinner("Extracting skills..."):
        extracted_skills = parse_pdf_for_skills(tmp_path)
    if extracted_skills:
        st.info(f"**Your Skills :** {extracted_skills}")
    else:
        st.warning("No recognizable skills found in the PDF.")

    # Predict clusters
    with st.spinner("Predicting clusters..."):
        results = predict_resume_clusters(tmp_path)

    if results:
        st.subheader("Top 3 Matching Clusters:")
        for cluster_id, label, score in results:
            st.write(f"**Cluster {cluster_id}: {label}**  (Score: {score:.3f})")
    else:
        st.warning("No matching clusters found.")