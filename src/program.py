import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pdfminer.high_level import extract_text
import pickle
# from src.job_scraper import get_job_listings  # Import the job scraping function
from src.skills import skilllist  # Import the skill list

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Data Cleaning function
def cleaning(text):
    text = re.sub(r"(https?://[^\s]+)", "", text)
    text = re.sub(r"(?:RT|cc|#\S+|@\S+)", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"[^\x00-\x7F]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Preprocessing function
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Function to extract skills from resume text
def extract_skills(text):
    extracted_skills = []
    for skill in skilllist:
        pattern = r'\b{}\b'.format(re.escape(skill.lower()))
        if re.search(pattern, text, re.IGNORECASE):
            extracted_skills.append(skill)
    return extracted_skills

# Prediction function for category based on resume
def predict_category_from_resume(pdf_path, model, vectorizer, label_encoder, skilllist):
    # Extract text from PDF
    print("Extracting text from PDF...")
    text = extract_text(pdf_path)
    if not text:
        print("No text extracted from PDF.")
        return "Error: No text extracted."

    print(f"Extracted text: {text[:500]}...")  # Print first 500 characters for debug

    cleaned_text = cleaning(text.strip())
    print(f"Cleaned text: {cleaned_text[:500]}...")  # Print cleaned text for debug

    text_skills = extract_skills(cleaned_text)
    print(f"Extracted skills: {text_skills}")

    text_skills_str = ' '.join(text_skills)
    print(f"Skills string: {text_skills_str}")

    # Vectorize the skills text
    text_vectorized = vectorizer.transform([text_skills_str])

    # Make a prediction using the loaded model
    prediction = model.predict(text_vectorized)
    print(f"Prediction: {prediction}")

    # Convert the predicted label back to its category
    predicted_category = label_encoder.inverse_transform(prediction)
    print(f"Predicted Category: {predicted_category[0]}")

    return predicted_category[0]

# Example usage
# Make sure the model, vectorizer, and label_encoder are loaded correctly in the Streamlit app or any calling script
if __name__ == "__main__":
    # Load the necessary components (assuming the files are in the correct locations)
    with open('./rf_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('./tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    with open('./label_encoder.pkl', 'rb') as le_file:
        label_encoder = pickle.load(le_file)

    # Test with a sample PDF
    pdf_path = './src/Resume.pdf'
    predicted_category = predict_category_from_resume(pdf_path, model, vectorizer, label_encoder, skilllist)
    print(f"Predicted category: {predicted_category}")
