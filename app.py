# Import necessary libraries 
import streamlit as st
import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
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
    df = pd.read_csv("processed_reviews.csv", encoding='ISO-8859-1')
    df.dropna(inplace=True)
    df['reviews'] = df[['Review_Title', 'Review']].agg('. '.join, axis=1).str.lstrip('. ')
    return df

# Sentiment analysis using VADER
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(review):
    return analyzer.polarity_scores(review)['compound']

# Word class categories
word_classes = {
    'safety': ['safety', 'secure', 'protection', 'safe', 'reliable'],
    'performance': ['performance', 'speed', 'acceleration', 'power', 'handling'],
    'comfort': ['comfort', 'convenient', 'cozy', 'soft', 'relax', 'spacious', 
        'seats', 'ergonomic', 'quiet', 'smooth', 'temperature', 
        'climate', 'ventilation', 'ride', 'luxurious', 'adjustable', 'supportive'],
    'design': ['design', 'style', 'appearance', 'look', 'aesthetic'],
    'economy': ['economy', 'fuel', 'efficient', 'mileage', 'cost']
}

def classify_user_input(input_text, word_classes):
    cleaned_input = preprocess_english_text(input_text, abbreviation_dict)
    class_counts = {cls: sum(word in cleaned_input for word in keywords) for cls, keywords in word_classes.items()}
    classified_class = max(class_counts, key=class_counts.get)
    return classified_class, class_counts

# Define a function to extract the year, brand, and name
def extract_year_brand_name(title):
    if isinstance(title, str):
        year = re.search(r'^\d{4}', title).group(0) if re.search(r'^\d{4}', title) else None
        brand = re.search(r'^\d{4}\s+(\w+)', title).group(1) if re.search(r'^\d{4}\s+(\w+)', title) else None
        car_name = re.search(r'^\d{4}\s+\w+\s+(.*)', title).group(1).strip() if re.search(r'^\d{4}\s+\w+\s+(.*)', title) else None
        return year, brand, car_name
    else:
        return None, None, None

# Extract components from the car model safely
def categorize_components_safe(car_model):
    try:
        stripped_model = car_model.strip('()')
        components = stripped_model.split()
        categorized = {'L': None, 'cyl': None, 'type': None, 'transmission': None, 'electric_DD': None}
        
        if stripped_model == 'electric DD':
            categorized['electric_DD'] = 'electric DD'
            categorized['L'] = 'not'
            categorized['cyl'] = 'not'
            categorized['type'] = 'not'
            categorized['transmission'] = 'not'
        elif len(components) >= 4:
            categorized['L'] = components[0]
            categorized['cyl'] = components[1]
            categorized['type'] = ' '.join(components[2:-1])
            categorized['transmission'] = components[-1]
            categorized['electric_DD'] = 'not'
        
        return categorized
    except Exception as e:
        return {'L': None, 'cyl': None, 'type': None, 'transmission': None, 'electric_DD': None}

# Count word classes in reviews
def count_word_classes(reviews, word_classes):
    class_counts = {cls: 0 for cls in word_classes.keys()}
    for review in reviews:
        words = review.split()
        for word in words:
            for cls, keywords in word_classes.items():
                if word in keywords:
                    class_counts[cls] += 1
    top_class = max(class_counts, key=class_counts.get)
    return top_class

# Count word classes with sentiment scores
def count_word_classes_with_sentiment(reviews, word_classes, sentiment_scores):
    class_counts = {cls: 0 for cls in word_classes.keys()}
    for i, review in enumerate(reviews):
        words = review.split()
        sentiment_score = sentiment_scores[i]
        for word in words:
            for cls, keywords in word_classes.items():
                if word in keywords:
                    class_counts[cls] += 1 * sentiment_score
    return class_counts

# Get word class counts for each car
def get_class_counts_by_car(df, word_classes):
    reviews_by_car = df.groupby('Car_Name').agg({
        'reviews_cleaned': lambda x: list(x),
        'vader_ss_normalize': lambda x: list(x)
    }).reset_index()

    class_counts_list = []
    car_names = []

    for i, row in reviews_by_car.iterrows():
        car_name = row['Car_Name']
        reviews = row['reviews_cleaned']
        sentiment_scores = row['vader_ss_normalize']
        class_counts = count_word_classes_with_sentiment(reviews, word_classes, sentiment_scores)
        class_counts_list.append(class_counts)
        car_names.append(car_name)
    
    class_counts_df = pd.DataFrame(class_counts_list, index=car_names)
    return class_counts_df

# Rank the cars for each category
def rank_cars_by_category(class_counts_df, category, top_n=5):
    ranked_cars = class_counts_df[category].sort_values(ascending=False).head(top_n)
    return ranked_cars

# UI Components
st.title("Sentiment Based Car Recommendation System")

# Load the dataset
df_reviews = load_data()

# Extract car year, brand, and name
df_reviews[['Car_Year', 'Car_Brand', 'Car_Name']] = df_reviews['Vehicle_Title'].apply(lambda x: pd.Series(extract_year_brand_name(x)))

# Apply the safe extraction to each row in the 'Vehicle_Title' column from df_reviews
extracted_data_safe = df_reviews['Vehicle_Title'].apply(
    lambda x: categorize_components_safe(re.search(r'\(.*\)', x).group(0)) 
    if pd.notnull(x) and re.search(r'\(.*\)', x) else {}
)
extracted_df_safe = pd.DataFrame(extracted_data_safe.tolist())
df_reviews = pd.concat([df_reviews, extracted_df_safe], axis=1)

# Preprocessing step
df_reviews['reviews_cleaned'] = df_reviews['reviews'].apply(lambda x: preprocess_english_text(x, abbreviation_dict))
df_reviews['vader_ss'] = df_reviews['reviews_cleaned'].apply(get_sentiment)
df_reviews['vader_ss_normalize'] = df_reviews['vader_ss'].apply(lambda x: 1 if x >= 0 else 0)

# Sidebar for manual or automatic input
option = st.sidebar.radio("Select Input Method", ("Manual Input", "Select from Options"))

# Manual input option
if option == "Manual Input":
    user_input = st.text_input("Describe your ideal car (e.g., safe, comfortable, etc.):")
    if user_input:
        classified_class, class_counts = classify_user_input(user_input, word_classes)
        st.write(f"Your input suggests you are looking for a car with a focus on **{classified_class}**.")
        
        # Calculate class counts for each car
        class_counts_df = get_class_counts_by_car(df_reviews, word_classes)
        
        # Rank cars based on user-selected category
        top_5_cars = rank_cars_by_category(class_counts_df, classified
