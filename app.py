import streamlit as st
import pickle
import re
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import random
import time
from PIL import Image
import base64

# Set page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon=":newspaper:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load the model
loaded_model = pickle.load(open('model.pkl', 'rb'))

# Initialize TF-IDF vectorizer
tfvect = TfidfVectorizer(max_df=0.7)

# Preprocess text
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and non-word characters
    words = nltk.word_tokenize(text.lower())  # Tokenization and lowercase conversion
    stop_words = set(stopwords.words('english'))
    stop_words.discard('no')  # Remove 'no' from the set of stopwords
    stop_words.discard('not')  # Remove 'not' from the set of stopwords
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Stopwords removal and lemmatization
    return ' '.join(words)

# Load and preprocess the training data
dataframe = pd.read_csv('New_train.csv')
ch = pd.read_csv("test.csv")
x = dataframe['news']
y = dataframe['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Function to predict fake news
def fake_news_det(news):
    input_data = preprocess_text(news)
    input_data = [input_data]
    tfid_x_train = tfvect.fit_transform(x_train)
    tfid_x_test = tfvect.transform(x_test)
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)[0]
    return prediction

# Streamlit app
def main():
    st.title("Fake News Detector")

    # Checkbox to randomly choose a news article from the dataset
    random_choice = st.checkbox("Randomly Choose a News Article", value=False)

    if random_choice:
        ip = ch['text']
        random_index = random.randint(0, len(ip)-1)
        news = ip.iloc[random_index]
        st.text_area("Randomly Chosen News Article:", news)

    # Text input for prediction
    if not random_choice:
        input_placeholder = st.empty()
        news = input_placeholder.text_area("Enter the news:", "")

    if st.button("Predict"):
        if news:
            with st.spinner("Predicting..."):
                prediction = fake_news_det(news)
                time.sleep(0.1)  # Simulate some processing time
            st.text(news)
            if prediction:
                st.success("Prediction: REAL")
            else:
                st.error("Prediction: FAKE")
        else:
            st.warning("Please enter the news for prediction.")

if __name__ == "__main__":
    main()
