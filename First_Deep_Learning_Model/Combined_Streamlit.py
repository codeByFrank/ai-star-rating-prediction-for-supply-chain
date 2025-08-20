import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

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

# Available models
MODEL_NAMES = [
    "deep_mlp_with_tf-idf",
    "lstm_model",
    "bilstm_with_attention",
    "cnn_model",
    "transformer_model",
    "hybrid_cnn-lstm"
]

# Initialize text preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


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


def load_mlp_specific_assets():
    """Load TF-IDF vectorizer and MLP metadata"""
    with open('api_models/deep_mlp_with_tf-idf/tfidf.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('api_models/deep_mlp_with_tf-idf/metadata.json', 'r') as f:
        mlp_metadata = json.load(f)
    return vectorizer, mlp_metadata


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


def preprocess_text(text, max_len=100):
    """Tokenize and preprocess text for models that don't use TF-IDF"""
    try:
        # Use the cached tokenizer if available
        tokenizer = artifacts.get('tokenizer')
        if tokenizer:
            sequence = tokenizer.texts_to_sequences([text])
            return sequence
    except:
        pass

    # Fallback simple tokenization (should match your training preprocessing)
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens
              if token not in stop_words and len(token) > 2]
    return [[len(token) for token in tokens[:max_len]]]  # Simplified example

def preprocess_for_mlp(text, vectorizer, num_features):
    """Process inputs specifically for MLP model"""
    # Transform text to TF-IDF
    text_features = vectorizer.transform([text]).toarray()

    # Combine with numerical features
    full_features = np.concatenate([text_features, num_features], axis=1)
    return full_features


# Load models and metadata
@st.cache_resource
def load_model_and_metadata(model_name):
    model_path = f"api_models/{model_name}/model.keras"
    metadata_path = f"api_models/{model_name}/metadata.json"

    model = load_model(model_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return model, metadata


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


def preprocess_inputs(model_name, review_text, numerical_features):
    if model_name == "deep_mlp_with_tf-idf":
        # Load TF-IDF vectorizer
        with open('api_models/deep_mlp_with_tf-idf/tfidf.pkl', 'rb') as f:
            vectorizer = pickle.load(f)

        # Process text and combine with numerical features
        return preprocess_for_mlp(review_text, vectorizer, numerical_features)
    else:
        # Text sequence processing for other models
        text_seq = preprocess_text(review_text, max_len=100)
        text_seq = tf.keras.preprocessing.sequence.pad_sequences(
            text_seq,
            maxlen=100,
            padding='post'
        )
        return [text_seq, numerical_features]


def show_prediction_page():
    st.header("Make a Prediction")

    # Model selection
    model_name = st.selectbox("Select Model", MODEL_NAMES)
    model, metadata = load_model_and_metadata(model_name)

    # Load MLP-specific assets if they exist
    try:
        tfidf_vectorizer, mlp_metadata = load_mlp_specific_assets()
    except:
        tfidf_vectorizer = None

    # Sample data
    sample_data = {
        #"ReviewText": "This product is amazing! It exceeded all my expectations.",
        "ReviewText": "This product is very good",
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

            # Prepare inputs for the selected model
            inputs = preprocess_inputs(model_name, review_text, num_features)

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

            # Make prediction
            try:
                if model_name == "deep_mlp_with_tf-idf":
                    # MLP expects single concatenated input
                    prediction = model.predict(inputs)
                else:
                    # Other models expect list of inputs
                    prediction = model.predict(inputs)

                # Process prediction results
                predicted_class = np.argmax(prediction, axis=1)[0] + 1
                probabilities = prediction[0]

                # Display results
                st.subheader("Prediction Results")
                st.write(f"Predicted Rating: {predicted_class} stars")

                # Probability distribution
                fig, ax = plt.subplots()
                sns.barplot(x=list(range(1, 6)), y=probabilities, ax=ax)
                ax.set_title("Rating Probability Distribution")
                ax.set_xlabel("Star Rating")
                ax.set_ylabel("Probability")
                st.pyplot(fig)

                # Confidence
                st.write(f"Confidence: {probabilities[predicted_class - 1] * 100:.1f}%")

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

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.write("Model input requirements:")
                if hasattr(model, 'input'):
                    st.json([inp.shape for inp in model.input])
                else:
                    st.json(model.input_shape)
                st.write("What you provided:")
                if isinstance(inputs, list):
                    st.json([x.shape for x in inputs])
                else:
                    st.json(inputs.shape)


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


def show_results():
    st.header("Model Evaluation Results")

    try:
        # Load the results
        with open('data/model_results.pkl', 'rb') as f:
            all_results = pickle.load(f)

        # Load test labels for ROC curves
        try:
            data = np.load('data/preprocessed_data.npz')
            y_test = data['y_test']
            has_roc_data = True
        except:
            y_test = None
            has_roc_data = False

        class_names = ["1", "2", "3", "4", "5"]

        # Create summary table
        st.subheader("Performance Summary")
        summary_data = []
        for result in all_results:
            summary_data.append({
                'Model': result['model_name'],
                'Accuracy': result['accuracy'],
                'F1-Weighted': result['f1_weighted'],
                'F1-Macro': result['f1_macro'],
                'AUC Score': result['auc_score']
            })

        summary_df = pd.DataFrame(summary_data).sort_values('F1-Weighted', ascending=False)
        st.dataframe(summary_df.style.format({
            'Accuracy': '{:.3f}',
            'F1-Weighted': '{:.3f}',
            'F1-Macro': '{:.3f}',
            'AUC Score': '{:.3f}'
        }))

        # Performance Metrics Bar Chart
        st.subheader("Performance Metrics Comparison")
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        metrics = ['Accuracy', 'F1-Weighted', 'F1-Macro', 'AUC Score']
        x = np.arange(len(summary_df))
        width = 0.2

        for i, metric in enumerate(metrics):
            ax1.bar(x + i * width, summary_df[metric], width, label=metric, alpha=0.8)

        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(summary_df['Model'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        plt.close(fig1)

        # Confusion Matrices
        st.subheader("Confusion Matrices")
        n_models = len(all_results)
        fig2, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

        if n_models == 1:
            axes = [axes]

        for ax, result in zip(axes, all_results):
            cm = result['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_title(result['model_name'])
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')

        st.pyplot(fig2)
        plt.close(fig2)

        # ROC Curves (only if we have y_test data and prediction probabilities)
        if has_roc_data:
            try:
                st.subheader("ROC Curves Comparison")
                fig3 = plt.figure(figsize=(10, 8))

                # Binarize the output
                y_true_bin = label_binarize(y_test, classes=np.arange(len(class_names)))

                # Check which models have prediction probabilities
                models_with_proba = [r for r in all_results if 'y_pred_proba' in r]

                if len(models_with_proba) > 0:
                    for result in models_with_proba:
                        y_pred_proba = result['y_pred_proba']

                        # Compute micro-average ROC curve and ROC area
                        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
                        roc_auc = auc(fpr, tpr)

                        plt.plot(fpr, tpr, label=f"{result['model_name']} (AUC = {roc_auc:.2f})")

                    plt.plot([0, 1], [0, 1], 'k--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Micro-average ROC Curve Comparison')
                    plt.legend(loc="lower right")
                    plt.grid(True, alpha=0.3)
                    st.pyplot(fig3)
                    plt.close(fig3)
                else:
                    st.warning("No models with prediction probabilities found for ROC curves")
            except Exception as e:
                st.warning(f"Could not generate ROC curves: {str(e)}")
        else:
            st.warning("ROC curves not available - test labels data not found")

    except FileNotFoundError:
        st.error("Results files not found. Please run model training first.")
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")


def show_intro_page():
    st.markdown("""

    <h1 style='color: #1f487e;font-size: 30px; text-align: center;'>Customer Satisfaction <br> (by Temu)</h1>
    <p style='text-align: center;'></p>
    <hr style="border: 2px solid #1f487e;">
    <p>
    In today’s competitive marketplace, <strong>customer satisfaction</strong> is no longer optional—it is a key driver of success.
    <p>
    Customer-generated <strong>reviews and ratings</strong> are powerful tools that help businesses understand what their clients expect, 
    value, and dislike. 
    <p>
    By leveraging this feedback and keeping a close eye on satisfaction levels, companies can adapt to 
    evolving needs, deliver a better user experience, and fine-tune their services—reducing customer churn in the process.             
    <p>
    When satisfaction improves, loyalty grows, often leading to stronger long-term revenue.
    <p>          
    The real challenge, however, is being able to analyze the large amount of customer feedback quickly and effectively.
    <p>

    <br>
    </p>
    """, unsafe_allow_html=True)


    # 3 columns
    col1, col2, col3 = st.columns([1, 2, 1])

    # Image in the midle
    with col2:
        st.image("satisfied-customer.png", width=600)

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

    ### Available Models
    The app supports multiple model architectures:
    - Deep MLP with TF-IDF
    - LSTM Model
    - BiLSTM with Attention
    - CNN Model
    - Transformer Model
    - Hybrid CNN-LSTM

    ### Data Source
    The model was trained on customer reviews from [Trustpilot/Temu](https://www.trustpilot.com/review/temu.com).
    """)

    st.subheader("Technical Details")
    st.write("""
    - **Preprocessing:** Text cleaning, tokenization, feature extraction
    - **Model Training:** 6 different architectures compared
    - **Evaluation Metrics:** Accuracy, F1-score, AUC-ROC
    """)

    st.subheader("Development Team")
    st.write("""
    - Data Scientists: [Mohamed ,Sebastian, Frank]
    - Machine Learning Engineer: [Mohamed ,Sebastian, Frank]
    """)


def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose a page",
                                ["Introduction", "Prediction", "Data Exploration", "Model Results", "About"])

    if app_mode == "Introduction":
        show_intro_page()
    elif app_mode == "Prediction":
        show_prediction_page()
    elif app_mode == "Data Exploration":
        show_data_exploration_page()
    elif app_mode == "Model Results":
        show_results()
    else:
        show_about_page()


if __name__ == "__main__":
    main()