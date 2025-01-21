import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Load the pre-trained model and vectorizer
@st.cache_resource
def load_model():
    with open("rf.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    return model

@st.cache_resource
def load_vectorizer():
    with open("tf.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return vectorizer

# Main function to build the Streamlit app
def main():
    st.title("Fake News Detection")
    st.write("Predict whether a piece of news is Fake or Real using a trained model.")

    # Load model and vectorizer
    model = load_model()
    vectorizer = load_vectorizer()

    # Input text area for user to enter news
    user_input = st.text_area("Enter the news content below:")

    if st.button("Predict"):
        if user_input.strip():
            # Transform the input text using the vectorizer
            transformed_input = vectorizer.transform([user_input])

            # Predict using the loaded model
            prediction = model.predict(transformed_input)

            # Show the prediction result
            if prediction[0] == 0:
                st.error("The news is predicted to be FAKE.")
            else:
                st.success("The news is predicted to be REAL.")
        else:
            st.warning("Please enter some text to predict.")

if __name__ == "__main__":
    main()
