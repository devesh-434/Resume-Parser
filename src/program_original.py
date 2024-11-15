
import pandas as pd
import numpy as np
import re
import nltk
import xgboost as xgb
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from pdfminer.high_level import extract_text
from io import StringIO
import pickle
from job_scraper import *
import warnings

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

df   = pd.read_csv('./data/UpdatedResumeDataSet.csv')

def cleaning(text):
    text = re.sub(r"(https?://[^\s]+)", "", text)
    text = re.sub(r"(?:RT|cc|#\S+|@\S+)", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"[^\x00-\x7F]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['cleaned'] = df['Resume'].apply(lambda x: cleaning(str(x)))
df['cleaned'] = df['cleaned'].str.lower()

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


skilllist = [
    "Java", "Python", "C++", "JavaScript", "C#", "PHP", "Ruby", "Go", "Swift", "Kotlin",
    "HTML", "CSS", "React", "Angular", "Vue.js", "Node.js", "Apache", "Nginx",
    "MySQL", "Oracle", "PostgreSQL", "MongoDB", "Cassandra", "Redis",
    "AWS", "Azure", "GCP","Bitcoin","Ethereum","Solidity","C"
    "Git", "version control systems", "Jenkins", "GitLab CI/CD","AUTO CAD","Fanuc Series"
    "TCP/IP", "network protocols", "routing", "firewalls",
    "Network security principles", "encryption", "firewalls", "IDS",
    "Windows (Server and Desktop)", "Linux (Ubuntu, Red Hat, CentOS)", "macOS",
    "Visual Studio", "PyCharm", "IntelliJ IDEA", "VS Code", "Sublime Text", "debuggers", "unit testing frameworks",
    "Requirements gathering", "process modeling", "stakeholder management", "data analysis",
    "Agile", "Waterfall", "MS Project", "Jira",
    "Lead generation", "market research", "sales strategy", "negotiation", "communication",
    "Supply chain management", "logistics", "Lean Six Sigma", "data-driven decision making",
    "Digital marketing", "social media marketing", "content marketing", "SEO", "SEM", "customer relationship management (CRM)",
    "SPIN Selling", "lead generation", "prospecting", "negotiation", "closing deals",
    "financial statements", "budgeting", "financial modeling", "risk management",
    "Recruitment", "onboarding", "performance management", "compensation and benefits", "employee relations",
    "Crisis communication", "media relations", "reputation management", "brand storytelling",
    "CRM systems (Salesforce, HubSpot)", "Asana", "Trello", "Tableau", "Power BI",
    "Communication", "negotiation", "problem-solving", "critical thinking", "leadership", "teamwork", "time management", "organization",
    "Design principles", "color theory", "typography", "layout", "Adobe Photoshop", "Illustrator", "InDesign",
    "UI design", "UX design", "Figma", "Sketch", "Adobe XD",
    "Adobe Premiere Pro", "motion graphics",
    "Social media marketing", "content creation (writing, editing, video)", "graphic design for social media",
    "Adobe Creative Suite", "Canva", "Piktochart",
    "Linear algebra", "calculus", "probability", "statistics", "hypothesis testing", "regression analysis",
    "Supervised learning (classification, regression)", "unsupervised learning (clustering)", "deep learning",
    "Data cleaning", "manipulation", "EDA", "Tableau", "Power BI)",
    "Spark", "Hadoop",
    "Python", "NumPy", "Pandas", "Scikit-learn", "R",
    "Jupyter Notebook", "RStudio", "AWS SageMaker", "Azure Machine Learning",
    "Microsoft Office Suite", "Word", "Excel", "PowerPoint",
    "Communication & Interpersonal Skills", "Problem-Solving & Critical Thinking", "Time Management & Organization",
    "Research & Analytical Skills", "Jira", "Asana", "Trello", "Cybersecurity Awareness"
]

def extract_skills(text):
    extracted_skills = []
    for skill in skilllist:
        pattern = r'\b{}\b'.format(re.escape(skill.lower()))
        if re.search(pattern, text, re.IGNORECASE):
            extracted_skills.append(skill)
    return extracted_skills


if 'Skills' not in df.columns:
    df['Skills'] = df['cleaned'].apply(extract_skills)

#print("Cleaned Text:", df['cleaned'].head())  # Print the first few entries of cleaned text
#print("Extracted Skills:", df['Skills'].head())  # Print extracted skills to check if they're empty


df['Skills'] = df['Skills'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
df_filtered = df[df['Skills'].notnull() & df['Category'].notnull()]



# Create skill_list and category_list from the filtered DataFrame
#skill_list = df_filtered['Skills'].tolist()
#category_list = df_filtered['Category'].tolist()


# Check if skill_list is empty or contains only empty strings
#if not skill_list or all(skill.strip() == '' for skill in skill_list):
    #raise ValueError("Skill list is empty or contains only stop words.")


tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df_filtered['Skills'])

df['Category'] = df['Category'].fillna('Unknown')

le = LabelEncoder()
y = le.fit_transform(df_filtered['Category'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# --- SVM Classifier ---
clf_svm = SVC()
clf_svm.fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_test)

# --- RandomForest Classifier ---
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

# --- Xgboost Classifier ---
xgb_clf = xgb.XGBClassifier()
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)

# Evaluation metrics
def evaluate_model(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    f1score = f1_score(y_test, y_pred, average='macro', zero_division=0)
    return acc, rec, precision, f1score

cv = 5  # Number of folds for cross-validation

# Define custom scorers to handle zero division warnings
accuracy_scorer = make_scorer(accuracy_score)
recall_scorer = make_scorer(recall_score, average='macro', zero_division=0)
precision_scorer = make_scorer(precision_score, average='macro', zero_division=0)
f1_scorer = make_scorer(f1_score, average='macro', zero_division=0)

# Cross-validation evaluation function
def cross_val_evaluation(model, X, y, cv):
    acc = np.mean(cross_val_score(model, X, y, cv=cv, scoring=accuracy_scorer))
    rec = np.mean(cross_val_score(model, X, y, cv=cv, scoring=recall_scorer))
    precision = np.mean(cross_val_score(model, X, y, cv=cv, scoring=precision_scorer))
    f1score = np.mean(cross_val_score(model, X, y, cv=cv, scoring=f1_scorer))
    return acc, rec, precision, f1score

# --- SVM Classifier ---
svm_acc_cv, svm_rec_cv, svm_precision_cv, svm_f1score_cv = cross_val_evaluation(clf_svm, X_train, y_train, cv)

# --- RandomForest Classifier ---
rf_acc_cv, rf_rec_cv, rf_precision_cv, rf_f1score_cv = cross_val_evaluation(rf_clf, X_train, y_train, cv)

# --- XGBoost Classifier ---
xgb_acc_cv, xgb_rec_cv, xgb_precision_cv, xgb_f1score_cv = cross_val_evaluation(xgb_clf, X_train, y_train, cv)

# Print the cross-validation performance comparison
print("\nCross-Validation Classifier Comparison:")
print(f"SVM (Cross-Validation) - Accuracy: {svm_acc_cv:.4f}, Recall: {svm_rec_cv:.4f}, Precision: {svm_precision_cv:.4f}, F1 Score: {svm_f1score_cv:.4f}")
print(f"RandomForest (Cross-Validation) - Accuracy: {rf_acc_cv:.4f}, Recall: {rf_rec_cv:.4f}, Precision: {rf_precision_cv:.4f}, F1 Score: {rf_f1score_cv:.4f}")
print(f"XGBoost (Cross-Validation) - Accuracy: {xgb_acc_cv:.4f}, Recall: {xgb_rec_cv:.4f}, Precision: {xgb_precision_cv:.4f}, F1 Score: {xgb_f1score_cv:.4f}")
# Evaluate SVM
svm_acc, svm_rec, svm_precision, svm_f1score = evaluate_model(y_test, y_pred_svm)
# Evaluate RandomForest
rf_acc, rf_rec, rf_precision, rf_f1score = evaluate_model(y_test, y_pred_rf)
# Evaluate XGBoost
xgb_acc, xgb_rec, xgb_precision, xgb_f1score = evaluate_model(y_test, y_pred_xgb)

# Print the performance comparison
print("Classifier Comparison:")
print(f"SVM - Accuracy: {svm_acc:.4f}, Recall: {svm_rec:.4f}, Precision: {svm_precision:.4f}, F1 Score: {svm_f1score:.4f}")
print(f"RandomForest - Accuracy: {rf_acc:.4f}, Recall: {rf_rec:.4f}, Precision: {rf_precision:.4f}, F1 Score: {rf_f1score:.4f}")
print(f"XGBoost - Accuracy: {xgb_acc:.4f}, Recall: {xgb_rec:.4f}, Precision: {xgb_precision:.4f}, F1 Score: {xgb_f1score:.4f}")



# Example prediction function
def prediction(pdf_path, vectorizer):
    text = extract_text(pdf_path)
    # Preprocess the extracted text
    cleaned_text = cleaning(text.strip())
    #print(cleaned_text)
    # Extract skills from preprocessed text
    text_skills = extract_skills(cleaned_text)

    # Join skills into a single string
    text_skills_str = ' '.join(text_skills)
    print(text_skills)

    text_vectorized = vectorizer.transform([text_skills_str])

    prediction = rf_clf.predict(text_vectorized)
    print(prediction)
    predicted_category = le.inverse_transform(prediction)
    print('The category is: ',predicted_category)
    if isinstance(predicted_category, (list, np.ndarray)):
        for category in predicted_category:
            transform(category)
    else:
        transform(predicted_category)


# Example usage
pdf_path = "./src/Resume.pdf"
print(prediction(pdf_path, tfidf))

