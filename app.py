import streamlit as st
from src.program import predict_category_from_resume  # Import the required functions
from src.skills import skilllist  # Import the skill list
from src.job_scraper import get_job_listings

import pickle

# Load the necessary components
with open('./rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('./tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('./label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

st.title('Resume Category Predictor & Job Listings')

# File uploader to upload PDF resumes
uploaded_file = st.file_uploader("Upload your resume (PDF)", type=['pdf'])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open(f"./temp_uploaded_resume.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Call the function to predict the category
    predicted_category = predict_category_from_resume(
        pdf_path="./temp_uploaded_resume.pdf", 
        model=model, 
        vectorizer=vectorizer, 
        label_encoder=label_encoder, 
        skilllist=skilllist
    )

    st.write(f"**Predicted Category:** {predicted_category}")

    # Fetch and display job listings based on the predicted category
    job_listings = get_job_listings(predicted_category)
    
    st.write("### Relevant Job Listings:")
    for job in job_listings:
        st.write(f"**Title:** {job['title']}")
        st.write(f"**Company:** {job['company']}")
        st.write("---")
