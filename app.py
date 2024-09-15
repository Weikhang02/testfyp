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

# Define a function to extract the year, brand, and name
def extract_year_brand_name(title):
    if isinstance(title, str):  # Ensure that the title is a string
        # Extract year (first four digits)
        year = re.search(r'^\d{4}', title).group(0) if re.search(r'^\d{4}', title) else None
        
        # Extract brand (first word after the year)
        brand = re.search(r'^\d{4}\s+(\w+)', title).group(1) if re.search(r'^\d{4}\s+(\w+)', title) else None
        
        # Extract car name (everything after the brand)
        car_name = re.search(r'^\d{4}\s+\w+\s+(.*)', title).group(1).strip() if re.search(r'^\d{4}\s+\w+\s+(.*)', title) else None
        
        return year, brand, car_name
    else:
        return None, None, None

# Define the function to extract components from the car model safely
def categorize_components_safe(car_model):
    try:
        # Remove the parentheses
        stripped_model = car_model.strip('()')
        
        # Split the components by spaces
        components = stripped_model.split()
        
        # Define a dictionary to hold categorized components, including electric DD
        categorized = {'L': None, 'cyl': None, 'type': None, 'transmission': None, 'electric_DD': None}
        
        # Check if the entire content is "electric DD"
        if stripped_model == 'electric DD':
            categorized['electric_DD'] = 'electric DD'
            categorized['L'] = 'not'
            categorized['cyl'] = 'not'
            categorized['type'] = 'not'
            categorized['transmission'] = 'not'
        elif len(components) >= 4:
            # Assign components to the respective categories
            categorized['L'] = components[0]
            categorized['cyl'] = components[1]
            categorized['type'] = ' '.join(components[2:-1])  # Everything in between
            categorized['transmission'] = components[-1]
            categorized['electric_DD'] = 'not'
        
        return categorized
    
    except Exception as e:
        # Return empty components if there's any error
        return {'L': None, 'cyl': None, 'type': None, 'transmission': None, 'electric_DD': None}

# UI Components
st.title("Car Recommendation System")

# Load the dataset
df_reviews = load_data()

# Extract car year, brand, and name
df_reviews[['Car_Year', 'Car_Brand', 'Car_Name']] = df_reviews['Vehicle_Title'].apply(lambda x: pd.Series(extract_year_brand_name(x)))

# Apply the safe extraction to each row in the 'Vehicle_Title' column from df_reviews
extracted_data_safe = df_reviews['Vehicle_Title'].apply(
    lambda x: categorize_components_safe(re.search(r'\(.*\)', x).group(0)) 
    if pd.notnull(x) and re.search(r'\(.*\)', x) else {}
)

# Convert the extracted data to a DataFrame
extracted_df_safe = pd.DataFrame(extracted_data_safe.tolist())

# Combine the original df_reviews DataFrame with the new extracted columns
df_reviews = pd.concat([df_reviews, extracted_df_safe], axis=1)

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
