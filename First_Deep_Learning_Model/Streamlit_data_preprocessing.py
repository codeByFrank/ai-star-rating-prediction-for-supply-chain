import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set page config
st.set_page_config(
    page_title="Customer Satisfaction Prediction",
    page_icon=":smiley:",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load preprocessing artifacts
@st.cache_resource
def load_artifacts():
    artifacts = {}
    try:
        with open('data/metadata.pkl', 'rb') as f:
            artifacts['metadata'] = pickle.load(f)
        with open('data/tokenizer.pkl', 'rb') as f:
            artifacts['tokenizer'] = pickle.load(f)
        with open('data/scaler.pkl', 'rb') as f:
            artifacts['scaler'] = pickle.load(f)
        with open('data/label_encoders.pkl', 'rb') as f:
            artifacts['label_encoders'] = pickle.load(f)
    except FileNotFoundError:
        st.error("Preprocessing artifacts not found. Please run the preprocessing notebook first.")
        st.stop()
    return artifacts


artifacts = load_artifacts()

# Initialize text preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text) or text == '':
        return ''

    # Convert to lowercase
    text = str(text).lower()

    # Remove URLs, email addresses
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)

    # Remove special characters and digits, keep only alphabets and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens
              if token not in stop_words and len(token) > 2]

    return ' '.join(tokens)


def preprocess_input(review_text, review_title, review_count, user_country):
    """Preprocess user input for prediction"""
    # Clean text
    clean_review = clean_text(review_text)
    clean_title = clean_text(review_title)
    combined_text = clean_review + ' ' + clean_title

    # Tokenize and pad text
    tokenizer = artifacts['tokenizer']
    sequence = tokenizer.texts_to_sequences([combined_text])
    padded_sequence = pad_sequences(sequence, maxlen=artifacts['metadata']['max_sequence_length'],
                                    padding='post', truncating='post')

    # Process numerical features
    country_encoded = artifacts['label_encoders']['UserCountry'].transform([user_country])[0]

    # Extract text features
    text_features = {
        'text_length': len(review_text),
        'word_count': len(review_text.split()),
        'avg_word_length': np.mean([len(word) for word in review_text.split()]) if review_text else 0,
        'exclamation_count': review_text.count('!'),
        'question_count': review_text.count('?'),
        'upper_case_ratio': sum(1 for c in review_text if c.isupper()) / len(review_text) if review_text else 0,
        'title_text_length': len(review_title),
        'title_word_count': len(review_title.split()),
        'title_avg_word_length': np.mean([len(word) for word in review_title.split()]) if review_title else 0,
        'title_exclamation_count': review_title.count('!'),
        'title_question_count': review_title.count('?'),
        'title_upper_case_ratio': sum(1 for c in review_title if c.isupper()) / len(review_title) if review_title else 0
    }

    # Create numerical feature array
    numerical_features = [
        review_count, country_encoded,
        text_features['text_length'], text_features['word_count'], text_features['avg_word_length'],
        text_features['exclamation_count'], text_features['question_count'], text_features['upper_case_ratio'],
        text_features['title_text_length'], text_features['title_word_count'], text_features['title_avg_word_length'],
        text_features['title_exclamation_count'], text_features['title_question_count'],
        text_features['title_upper_case_ratio']
    ]

    # Scale numerical features
    scaled_numerical = artifacts['scaler'].transform([numerical_features])

    return padded_sequence, scaled_numerical


# Streamlit app layout
def main():
    st.title("Customer Satisfaction Prediction")
    st.markdown("""
    This app predicts customer satisfaction ratings (1-5 stars) based on review text and other features.
    """)

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose a page",
                                ["Prediction", "Data Exploration", "About"])

    if app_mode == "Prediction":
        show_prediction_page()
    elif app_mode == "Data Exploration":
        show_data_exploration_page()
    else:
        show_about_page()


def show_prediction_page():
    st.header("Make a Prediction")

    # Load sample data
    sample_data = {
        "ReviewText": "This product is amazing! It exceeded all my expectations.",
        "ReviewTitle": "Best purchase ever",
        "ReviewCount": 5,
        "UserCountry": "US"
    }

    # User input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            review_text = st.text_area("Review Text", value=sample_data["ReviewText"], height=150)
            review_count = st.number_input("Review Count", min_value=1, value=sample_data["ReviewCount"])

        with col2:
            review_title = st.text_input("Review Title", value=sample_data["ReviewTitle"])
            user_country = st.selectbox("User Country",
                                        options=artifacts['label_encoders']['UserCountry'].classes_,
                                        index=0)

        submitted = st.form_submit_button("Predict Rating")

    if submitted:
        with st.spinner("Processing your review..."):
            # Preprocess input
            text_seq, num_features = preprocess_input(
                review_text, review_title, review_count, user_country
            )

            # Display cleaned text
            st.subheader("Preprocessed Text")
            cleaned_review = clean_text(review_text)
            cleaned_title = clean_text(review_title)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Review**")
                st.write(review_text)
            with col2:
                st.markdown("**Cleaned Review**")
                st.write(cleaned_review)

            # Make prediction (placeholder - you'll need to load your actual model)
            st.subheader("Prediction Results")

            # Mock prediction - replace with your actual model
            pred_probs = np.random.dirichlet(np.ones(5), size=1)[0]
            pred_class = np.argmax(pred_probs) + 1

            # Display results
            st.markdown(f"**Predicted Rating:** {pred_class} stars")

            # Show probability distribution
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=list(range(1, 6)), y=pred_probs, palette="Blues_d", ax=ax)
            ax.set_title("Prediction Probability Distribution")
            ax.set_xlabel("Rating (stars)")
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)
            st.pyplot(fig)

            # Show feature importance (placeholder)
            st.subheader("Key Influencing Factors")
            factors = [
                "Positive sentiment in review",
                "Review length",
                "Use of exclamation marks",
                "User's review history"
            ]
            for i, factor in enumerate(factors, 1):
                st.markdown(f"{i}. {factor}")


def show_data_exploration_page():
    st.header("Data Exploration")

    # Load sample data (in a real app, you'd load your actual data)
    st.subheader("Dataset Overview")
    st.write("""
    The dataset contains customer reviews with ratings from 1 to 5 stars.
    Below is a sample of the data distribution and features.
    """)

    # Show target distribution
    st.subheader("Rating Distribution")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Mock data - replace with your actual data
    rating_counts = pd.Series({
        1: 7082,
        2: 850,
        3: 644,
        4: 1099,
        5: 3919
    })

    rating_counts.sort_index().plot(kind='bar', ax=ax[0])
    ax[0].set_title('Review Count by Rating')
    ax[0].set_xlabel('Rating')
    ax[0].set_ylabel('Count')

    rating_counts.sort_index().plot(kind='pie', autopct='%1.1f%%', ax=ax[1])
    ax[1].set_title('Rating Distribution (%)')
    ax[1].set_ylabel('')

    st.pyplot(fig)

    # Show feature distributions
    st.subheader("Feature Distributions")
    feature = st.selectbox("Select a feature to visualize",
                           artifacts['metadata']['feature_columns'])

    # Mock feature distribution - replace with your actual data
    if "length" in feature or "count" in feature:
        dist_data = np.random.poisson(10, 1000)
    else:
        dist_data = np.random.normal(0, 1, 1000)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(dist_data, bins=30, kde=True, ax=ax)
    ax.set_title(f"Distribution of {feature}")
    st.pyplot(fig)


def show_about_page():
    st.header("About This App")
    st.markdown("""
    ### Customer Satisfaction Prediction App

    This application predicts customer satisfaction ratings (1-5 stars) based on:
    - Review text content
    - Review title
    - User metadata (country, review count)
    - Text features (length, sentiment, punctuation, etc.)

    ### How It Works
    1. The app preprocesses the input text (cleaning, tokenization)
    2. Extracts numerical features from the text and user data
    3. Uses a trained deep learning model to predict the rating

    ### Model Architecture
    The prediction model uses a hybrid architecture combining:
    - Text processing with LSTM/Transformer layers
    - Numerical feature processing with dense layers

    ### Data Source
    The model was trained on customer reviews from [Temu](https://www.temu.com/).
    """)

    st.subheader("Technical Details")
    st.write("""
    - **Preprocessing:** Text cleaning, tokenization, feature extraction
    - **Model Training:** 5 different architectures compared
    - **Evaluation Metrics:** Accuracy, F1-score, AUC-ROC
    """)

    st.subheader("Development Team")
    st.write("""
    - Data Scientist: [Your Name]
    - Machine Learning Engineer: [Your Name]
    """)


if __name__ == "__main__":
    main()