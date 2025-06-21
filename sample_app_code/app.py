#

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Page setup - must be the first Streamlit command
st.set_page_config(layout="wide")

# Initialize BERT model
@st.cache_resource
def init_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def main():
    model = init_model()

    st.title("📄 Resume-Job Matcher")
    st.markdown("Match resumes with jobs using AI-powered semantic matching")

    # Create two columns for input
    col1, col2 = st.columns(2)

    # Resume inputs
    with col1:
        st.subheader("Resume Details")
        skills = st.text_area("Skills", "Python, Machine Learning, Data Analysis")
        experience = st.text_area("Experience", "5 years software development")
        education = st.text_area("Education", "MS Computer Science")
        resume_industry = st.selectbox(
            "Resume Industry",
            ["Technology", "Healthcare", "Finance", "Education", "Other"],
            key="resume_industry"
        )

    # Job inputs
    with col2:
        st.subheader("Job Details")
        requirements = st.text_area("Requirements", "Python developer with ML experience")
        description = st.text_area("Description", "Looking for an ML engineer")
        job_industry = st.selectbox(
            "Job Industry",
            ["Technology", "Healthcare", "Finance", "Education", "Other"],
            key="job_industry"
        )

    def get_embedding(text):
        return model.encode(text, convert_to_tensor=True)

    def calculate_similarity(resume_text, job_text):
        # Combine all resume fields and job fields
        resume_combined = f"{skills} {experience} {education}"
        job_combined = f"{requirements} {description}"
        
        # Get embeddings
        resume_embedding = get_embedding(resume_combined)
        job_embedding = get_embedding(job_combined)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(
            resume_embedding.cpu().numpy().reshape(1, -1),
            job_embedding.cpu().numpy().reshape(1, -1)
        )[0][0]
        
        return similarity

    def get_match_quality(similarity_score):
        if similarity_score >= 0.7:
            return "Strong Match 🌟"
        elif similarity_score >= 0.5:
            return "Moderate Match ⭐"
        else:
            return "Low Match 📝"

    if st.button("Find Matches"):
        # Check if industries match
        industry_match = resume_industry == job_industry
        
        # Calculate similarity
        similarity_score = calculate_similarity(
            f"{skills} {experience} {education}",
            f"{requirements} {description}"
        )
        
        # Display results
        st.header("Match Results")
        
        # Display the similarity score with a progress bar
        st.metric("Similarity Score", f"{similarity_score:.2%}")
        st.progress(float(similarity_score))
        
        # Display match quality
        match_quality = get_match_quality(similarity_score)
        st.subheader(f"Match Quality: {match_quality}")
        
        # Industry match indicator
        if industry_match:
            st.success("✅ Industries Match!")
        else:
            st.warning("⚠️ Industries Differ")
        
        # Create comparison table
        comparison_data = {
            "Aspect": ["Skills", "Experience", "Education", "Industry"],
            "Resume": [skills, experience, education, resume_industry],
            "Job": [requirements, description, "", job_industry]
        }
        
        st.table(pd.DataFrame(comparison_data))

if __name__ == "__main__":
    main()