import joblib
import numpy as np
from utils import comma_tokenizer
from sklearn.metrics.pairwise import cosine_similarity
from resume_parser import parse_pdf_for_skills

# Define tokenizer to avoid lambda (so we can pickle)
# def custom_tokenizer(text):
#     return [token.strip() for token in text.split(',') if token.strip()]

# Load models
vectorizer = joblib.load("skill_vectorizer.pkl")
kmeans = joblib.load("skill_kmeans_model.pkl")
cluster_labels = joblib.load("cluster_labels.pkl")  # dict: cluster_id → domain

# Predict top 3 matching clusters from resume PDF
def predict_resume_clusters(pdf_path, skill_list_file="unique_skills.txt"):
    # Extract skills
    extracted_skills = parse_pdf_for_skills(pdf_path, skill_list_file)
    if not extracted_skills:
        print("❌ No recognizable skills found.")
        return []

    print("✅ Extracted Skills:", extracted_skills)

    # Vectorize skills
    vec = vectorizer.transform([extracted_skills])

    # Compare with cluster centers
    similarities = cosine_similarity(vec, kmeans.cluster_centers_)[0]
    top_indices = np.argsort(similarities)[::-1][:3]

    results = []
    for idx in top_indices:
        label = cluster_labels.get(idx, f"Cluster {idx}")
        score = similarities[idx]
        results.append((idx, label, score))

    return results

# pdf_path = "Resume.pdf"
# # ✅ Run prediction
# results = predict_resume_clusters(pdf_path)

# # 🎯 Show result
# if results:
#     print("\n🔍 Top 3 Matching Clusters Based on Your Resume:")
#     for cluster_id, label, score in results:
#         print(f"✅ Cluster {cluster_id}: {label}  (Score: {score:.3f})")
# else:
#     print("\n⚠️ No matching clusters found.")