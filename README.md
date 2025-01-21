
# NLP Text Classification Project

## Overview
This project focuses on text classification using Natural Language Processing (NLP) techniques. The primary goal is to build a web application that predicts whether a given news article is **Fake** or **Real** using a pre-trained machine learning model.

The application is implemented using **Streamlit**, providing an interactive interface for users to input news content and receive predictions.

---

## Features
- **Fake News Detection**: Input news content to determine if it's fake or real.
- **Pre-trained Model**: Uses a Random Forest Classifier trained on labeled news datasets.
- **User-friendly Interface**: Built with Streamlit for simplicity and ease of use.

---

## Project Structure
- **app.py**: Main Streamlit application file.
- **rf.pkl**: Serialized pre-trained Random Forest Classifier model.
- **tf.pkl**: Serialized TF-IDF vectorizer for text transformation.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/nlp-text-classification.git
   ```

2. Navigate to the project directory:
   ```bash
   cd nlp-text-classification
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and go to:
   
3. Enter the news content in the text area provided and click **Predict** to see the result.

---

## Dependencies
- **Streamlit**: Interactive UI framework.
- **pandas**: Data manipulation.
- **numpy**: Numerical computations.
- **scikit-learn**: Machine learning utilities.
- **NLTK**: Natural language processing.

---

## Model and Data
- **Random Forest Classifier**: A robust model for classification tasks.
- **TF-IDF Vectorizer**: Converts text data into numerical features for model input.

---

## Example
1. Enter the news content:
   ```
   Breaking: Scientists discover a new exoplanet in the habitable zone.
   ```

2. Click **Predict**.

3. Get the result:
   - **REAL**: The news is genuine.
   - **FAKE**: The news is fabricated.

---

## File Details
### app.py
Contains the following key functionalities:
- Loads the pre-trained model and vectorizer using `pickle`.
- Prepares the user interface using Streamlit.
- Processes user input and provides predictions.

### rf.pkl
Serialized Random Forest Classifier trained on labeled datasets for Fake News detection.

### tf.pkl
Serialized TF-IDF Vectorizer for transforming textual data into numerical format.

---




---

### Note
Ensure that `rf.pkl` and `tf.pkl` are available in the project directory before running the application.

