# Import necessary libraries
import streamlit as st
import pandas as pd
import nltk
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

# Download nltk resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Define preprocessing functions
stop_words = set(stopwords.words('english'))
abbreviation_dict = {
    'wif': 'with',
    'hv': 'have',
    'EV': 'Electric Vehicle',
    'shld': 'should',
    'btw': 'by the way',
    'bc': 'because'
}

def replace_abbreviations(text, abbreviation_dict):
    words = text.split()
    replaced_text = ' '.join([abbreviation_dict.get(word, word) for word in words])
    return replaced_text

def preprocess_english_text(text, abbreviation_dict):
    if not isinstance(text, str):
        return ""
    text = replace_abbreviations(text, abbreviation_dict)
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens])

# Load data
@st.cache
def load_data():
    df = pd.read_csv("Review(Latest).csv", encoding='ISO-8859-1')
    df.dropna(inplace=True)
    df['reviews'] = df[['Review_Title', 'Review']].agg('. '.join, axis=1).str.lstrip('. ')
    return df

# Sentiment analysis using VADER
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(review):
    return analyzer.polarity_scores(review)['compound']

# Recommendation system function
word_classes = {
    'safety': ['safety', 'secure', 'protection', 'safe', 'reliable'],
    'performance': ['performance', 'speed', 'acceleration', 'power', 'handling'],
    'comfort': ['comfort', 'convenient', 'cozy', 'spacious', 'seats'],
    'design': ['design', 'style', 'appearance', 'look', 'aesthetic'],
    'economy': ['economy', 'fuel', 'efficient', 'mileage', 'cost']
}

def classify_user_input(input_text, word_classes):
    cleaned_input = preprocess_english_text(input_text, abbreviation_dict)
    class_counts = {cls: sum(word in cleaned_input for word in keywords) for cls, keywords in word_classes.items()}
    classified_class = max(class_counts, key=class_counts.get)
    return classified_class, class_counts

# UI Components
st.title("Car Recommendation System")

# Load the dataset
df_reviews = load_data()

# Preprocessing step
df_reviews['reviews_cleaned'] = df_reviews['reviews'].apply(lambda x: preprocess_english_text(x, abbreviation_dict))
df_reviews['vader_ss'] = df_reviews['reviews_cleaned'].apply(get_sentiment)
df_reviews['vader_ss_normalize'] = df_reviews['vader_ss'].apply(lambda x: 1 if x >= 0 else 0)

# User input for car preference
user_input = st.text_input("Describe your ideal car (e.g., safe, comfortable, etc.):")

if user_input:
    classified_class, class_counts = classify_user_input(user_input, word_classes)
    st.write(f"Your input suggests you are looking for a car with a focus on **{classified_class}**.")

    # Example recommendation logic based on sentiment analysis and classification
    recommended_cars = df_reviews[df_reviews['vader_ss_normalize'] == 1].head(5)
    
    st.write("Here are the top 5 cars matching your preferences:")
    
    for index, row in recommended_cars.iterrows():
        st.write(f"**{row['Car_Brand']} {row['Car_Name']} ({row['Car_Year']})** - {row['Price']}")

# Visualize the word class counts
if user_input:
    st.write("Word Class Counts in your input:")
    st.bar_chart(pd.DataFrame(class_counts.values(), index=class_counts.keys(), columns=['Counts']))
