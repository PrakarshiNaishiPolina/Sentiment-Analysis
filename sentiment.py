import streamlit as st
import nltk
import string
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis App", 
    page_icon="🚀",  
    layout="centered",  
    initial_sidebar_state="collapsed", 
)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Ensure stopwords are properly loaded
stop_words = set(stopwords.words('english'))

# Preprocess function
def clean_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

# Load dataset and preprocess
categories = ['rec.autos', 'sci.med']
data = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))

X = [clean_text(text) for text in data.data]
y = [1 if 'rec.autos' in data.target_names[label] else 0 for label in data.target]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline = Pipeline([('tfidf', TfidfVectorizer(max_features=5000)), ('clf', MultinomialNB())])
pipeline.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

# Streamlit App
st.title("Sentiment Analysis App 🚀")
st.write("Enter text to analyze sentiment.")

text = st.text_area("Enter your text:")

if st.button("Analyze"):
    if text:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        
        # Clean the input text before prediction
        cleaned_text = clean_text(text)
        
        prediction = model.predict([cleaned_text])[0]
        polarity = TextBlob(text).sentiment.polarity
        sentiment = "Positive 😊" if polarity > 0 else "Negative 😠" if polarity < 0 else "Neutral 😐"

        st.write(f"**Predicted Category:** {'Automobile' if prediction == 1 else 'Medical'}")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Polarity Score:** {polarity:.2f}")
    else:
        st.warning("Please enter text!")

st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
        font-family: 'Arial', sans-serif;
    }
    .stTextArea label {
        font-size: 18px;
        font-weight: bold;
        color: #333;
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        font-size: 16px;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 8px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stAlert {
        border-radius: 10px;
        padding: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

