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
from PIL import Image
import os

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
        car_name = re.search(r'^\d{4}\s+\w+\s+([^\(]+)', title).group(1).strip() if re.search(r'^\d{4}\s+\w+\s+([^\(]+)', title) else None
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
            categorized['L'] = 'no'
            categorized['cyl'] = 'no'
            categorized['type'] = 'no'
            categorized['transmission'] = 'no'
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
def calcMissingRowCount(df):
    # summing up the missing values (column-wise) and displaying fraction of NaNs
    return df.isnull().sum()

# UI Components
st.title("Sentiment Based Car Recommendation System")

# Load the dataset
df = load_data()
df_reviews= df.dropna()
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
#option = st.sidebar.radio("Select Input Method", ("Manual Input", "Select from Options"))

# Define a function to extract brand, price range, and other features from user input
def extract_car_features(input_text):
    brand_pattern = re.search(r'\b(Dodge|FIAT|Ford|HUMMER|Hyundai|INFINITI|Jaguar|Kia|Lamborghini|Lexus|BMW|Audi|Porsche|Volkswagen|Toyota|Honda|Tesla|Bentley|GMC|Ford)\b', input_text, re.IGNORECASE)
    price_pattern = re.search(r'price range of (\d+)-(\d+)', input_text)
    engine_pattern = re.search(r'(\d\.\d)L', input_text)
    cyl_pattern = re.search(r'(\d)cyl', input_text)
    type_pattern = re.search(r'(Turbo|Hybrid|Electric|Diesel|Petrol|etc)', input_text, re.IGNORECASE)
    transmission_pattern = re.search(r'(6M|5M|Automatic|Manual|CVT|etc)', input_text, re.IGNORECASE)

    return {
        'Car_Brand': brand_pattern.group(0) if brand_pattern else None,
        'Price_Min': int(price_pattern.group(1)) if price_pattern else None,
        'Price_Max': int(price_pattern.group(2)) if price_pattern else None,
        'L': engine_pattern.group(0) if engine_pattern else None,
        'cyl': cyl_pattern.group(0) if cyl_pattern else None,
        'type': type_pattern.group(0) if type_pattern else None,
        'transmission': transmission_pattern.group(0) if transmission_pattern else None,
    }

def get_car_image(car_name):
    # Convert car name to a valid file name (replace spaces with underscores)
    #file_name = car_name.replace(" " , "_") + ".jpg"
    file_name="Caliber_SRT4_SRT4_4dr_Wagon.jpg"
    # Define the base path to your local images folder
    image_base_path = "./Car Image"  # Adjust the path as needed
    
    # Construct the full local image path
    image_path = os.path.join(image_base_path, file_name)
    
    try:
        # Check if the image exists in the specified path
        if os.path.exists(image_path):
            return image_path
        else:
            # If image not found, return a placeholder image path
            return os.path.join(image_base_path, "no image available.jpg")  # Add "no_image_available.jpg" in your folder
    except Exception as e:
        st.write(f"Error loading image: {e}")
        return os.path.join(image_base_path, "no image available.jpg")

# Ensure 'Price' is numeric and handle any NaN values
#df_reviews['Price'] = pd.to_numeric(df_reviews['Price'], errors='coerce')  # Convert Price to numeric, invalid parsing will be set as NaN

# Modify the manual input option to include car features

df_reviews['Price'] = df_reviews['Price'].fillna(0)
# st.write(calcMissingRowCount(df_reviews['Price']))
# Modify the manual input option to include car features and logic for filtering or showing top 5 cars
# Sidebar for user input options: either "I know my preferences" or "I need recommendations"
# Sidebar for user input options: either "I know my preferences" or "I need recommendations"
option = st.sidebar.radio("Choose how you'd like to proceed:", 
                          ("I know my preferences", "I need top 5 recommendations"))

# If the user knows their preferences
if option == "I know my preferences":
    st.write("Please specify your car preferences below:")
    
    # Let the user describe their ideal car
    user_input = st.text_input("Describe your ideal car (e.g., brand, price range of (xxx to xxx), L, cyl, type or transmission):")
    
    if user_input:
        # Extract car features from user input
        car_features = extract_car_features(user_input)
        
        # Show extracted features
        st.write(f"Extracted Car Features: {car_features}")
        
        # Filter the dataset based on the extracted car features
        filtered_df = df_reviews.copy()
        if car_features['Car_Brand']:
            filtered_df = filtered_df[filtered_df['Car_Brand'].fillna('').str.contains(car_features['Car_Brand'], case=False)]
        if car_features['Price_Min'] and car_features['Price_Max']:
            filtered_df = filtered_df[(filtered_df['Price'] >= car_features['Price_Min']) & (filtered_df['Price'] <= car_features['Price_Max'])]
        if car_features['L']:
            filtered_df = filtered_df[filtered_df['L'].fillna('').str.contains(car_features['L'], case=False)]
        if car_features['cyl']:
            filtered_df = filtered_df[filtered_df['cyl'].fillna('').str.contains(car_features['cyl'], case=False)]
        if car_features['type']:
            filtered_df = filtered_df[filtered_df['type'].fillna('').str.contains(car_features['type'], case=False)]
        if car_features['transmission']:
            filtered_df = filtered_df[filtered_df['transmission'].fillna('').str.contains(car_features['transmission'], case=False)]
        
        # If specific features are provided, find the car with the highest sentiment score
        if not filtered_df.empty:
            # Calculate sentiment score and sort by highest score
            filtered_df['sentiment_score'] = filtered_df['vader_ss']
            highest_sentiment_car = filtered_df.loc[filtered_df['sentiment_score'].idxmax()]

            # Display the car with the highest sentiment score
            st.write("Car matching your criteria with the highest sentiment score:")
            
            # Get the car name and retrieve the image
            car_name = highest_sentiment_car['Car_Name']
            car_image_path = get_car_image(car_name)
            
            # Check if the image is valid before displaying
            #if car_image_path is not None:
            #    st.image(car_image_path, caption=car_name, use_column_width=True)
            #else:
            #    st.write("Image could not be loaded.")
            # Display the image along with car details
            st.image(car_image_path, caption=car_name, use_column_width=True)

            if (highest_sentiment_car['L']=="no") and (highest_sentiment_car['cyl']=="no") and (highest_sentiment_car['type']=="no") and (highest_sentiment_car['transmission']=="no"):
                # Create a DataFrame to display car details in table format
                car_details = {
                    "Car Year": [highest_sentiment_car['Car_Year']],
                    "Car Brand": [highest_sentiment_car['Car_Brand']],
                    "Car Name": [highest_sentiment_car['Car_Name']],
                    "Electric Drive": [highest_sentiment_car['electric_DD']],
                    "Price": [highest_sentiment_car['Price']]
                }
                
                # Convert the dictionary to a DataFrame
                car_details_df = pd.DataFrame(car_details)
            
                # Display the DataFrame as a table
                st.table(car_details_df)
            else:
                 # Create a DataFrame to display car details in table format
                 car_details = {
                     "Car Year": [highest_sentiment_car['Car_Year']],
                     "Car Brand": [highest_sentiment_car['Car_Brand']],
                     "Car Name": [highest_sentiment_car['Car_Name']],
                     "Engine": [highest_sentiment_car['L']],
                     "Cylinders": [highest_sentiment_car['cyl']],
                     "Type": [highest_sentiment_car['type']],
                     "Transmission": [highest_sentiment_car['transmission']],
                     "Price": [highest_sentiment_car['Price']]
                 }
                
                 # Convert the dictionary to a DataFrame
                 car_details_df = pd.DataFrame(car_details)

                 # Display the DataFrame as a table
                 st.table(car_details_df)
        # If no cars match the features, provide a fallback message
        else:
            st.write("No cars found matching your exact preferences.")
        
# If the user doesn't know their preferences
elif option == "I need top 5 recommendations":
    st.write("We will show you the top 5 recommended cars based on sentiment analysis.")
    
    # Let the user specify if they are looking for any particular feature, if desired (optional)
    user_input = st.text_input("If you have any particular preferences (e.g., comfort, performance), enter them here:")
    
    if user_input:
        # Proceed with word class classification if the user enters preferences
        classified_class, class_counts = classify_user_input(user_input, word_classes)
        st.write(f"Your input suggests you are looking for a car with a focus on **{classified_class}**.")
    
        # Calculate class counts for each car
        class_counts_df = get_class_counts_by_car(df_reviews, word_classes)
    
        # Rank cars based on the classified word class or default to 'comfort'
        category = classified_class if user_input else 'comfort'  # Default to 'comfort' if no input
        top_5_cars = rank_cars_by_category(class_counts_df, category, top_n=5)
        
        # Match top 5 cars by car name to retrieve the relevant details
        top_5_car_details = df_reviews[df_reviews['Car_Name'].isin(top_5_cars.index)]
        
        # Remove duplicate entries by keeping the first occurrence of each car name
        top_5_car_details = top_5_car_details.drop_duplicates(subset=['Car_Name'], keep='first')
        
        # Display the top 5 cars
        st.write(f"Here are the top 5 cars based on **{category}**:")
        
        for index, row in top_5_car_details.iterrows():
            # Get the car name and retrieve the image
            car_name = row['Car_Name']
            car_image_path = get_car_image(car_name)
            
            # Display the image along with car details
            st.image(car_image_path, caption=car_name, use_column_width=True)

            if (row['L']=="no") and (row['cyl']=="no") and (row['type']=="no") and (row['transmission']=="no"):
                # Create a DataFrame to display car details in table format
                car_details = {
                    "Car Year": [row['Car_Year']],
                    "Car Brand": [row['Car_Brand']],
                    "Car Name": [row['Car_Name']],
                    "Electric Drive": [row['electric_DD']],
                    "Price": [row['Price']]
                }
            
                # Convert the dictionary to a DataFrame
                car_details_df = pd.DataFrame(car_details)
            
                # Display the DataFrame as a table
                st.table(car_details_df)
            else:
                 # Create a DataFrame to display car details in table format
                 car_details = {
                     "Car Year": [row['Car_Year']],
                     "Car Brand": [row['Car_Brand']],
                     "Car Name": [row['Car_Name']],
                     "Engine": [row['L']],
                     "Cylinders": [row['cyl']],
                     "Type": [row['type']],
                     "Transmission": [row['transmission']],
                     "Price": [row['Price']]
                 }
                
                 # Convert the dictionary to a DataFrame
                 car_details_df = pd.DataFrame(car_details)

                 # Display the DataFrame as a table
                 st.table(car_details_df)
