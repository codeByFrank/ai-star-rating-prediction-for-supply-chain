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


def load_mlp_specific_assets():
    """Load TF-IDF vectorizer and MLP metadata"""
    with open('api_models/deep_mlp_with_tf-idf/tfidf.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('api_models/deep_mlp_with_tf-idf/metadata.json', 'r') as f:
        mlp_metadata = json.load(f)
    return vectorizer, mlp_metadata


def preprocess_for_mlp(text, vectorizer, num_features):
    """Process inputs specifically for MLP model"""
    # Transform text to TF-IDF
    text_features = vectorizer.transform([text]).toarray()

    # Combine with numerical features
    full_features = np.concatenate([text_features, num_features], axis=1)
    return full_features

# Load models and metadata
def load_model_and_metadata(model_name):
    model_path = f"api_models/{model_name}/model.keras"
    metadata_path = f"api_models/{model_name}/metadata.json"

    model = load_model(model_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return model, metadata


# Available models
MODEL_NAMES = [
    "deep_mlp_with_tf-idf",
    "lstm_model",
    "bilstm_with_attention",
    "cnn_model",
    "transformer_model",
    "hybrid_cnn-lstm"
]


# Text preprocessing (simplified - should match your training preprocessing)
def preprocess_text(text, max_len=100):
    # Load tokenizer if available
    try:
        with open('api_models/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer.texts_to_sequences([text])
    except:
        # Fallback simple tokenization
        return [[len(word) for word in text.split()[:max_len]]]  # Simplified example

        # Main app
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

def main():
    st.title("Customer Satisfaction Prediction")
    st.write("Predict star ratings (1-5) from customer reviews")

    # Model selection
    model_name = st.selectbox("Select Model", MODEL_NAMES)
    model, metadata = load_model_and_metadata(model_name)

    # Load MLP-specific assets if they exist
    try:
        tfidf_vectorizer, mlp_metadata = load_mlp_specific_assets()
    except:
        tfidf_vectorizer = None

    # Input form
    with st.form("prediction_form"):
        st.subheader("Review Information")

        # Text inputs
        review_text = st.text_area("Review Text", "The product was great!")
        review_title = st.text_input("Review Title", "Awesome product")

        # Numerical features
        st.subheader("Numerical Features")
        col1, col2 = st.columns(2)

        with col1:
            review_count = st.number_input("Review Count", min_value=0, value=1)
            text_length = st.number_input("Text Length", min_value=0, value=len(review_text))
            word_count = st.number_input("Word Count", min_value=0, value=len(review_text.split()))
            exclamation_count = st.number_input("Exclamation Count", min_value=0,
                                                value=review_text.count('!'))

        with col2:
            question_count = st.number_input("Question Count", min_value=0,
                                             value=review_text.count('?'))
            upper_case_ratio = st.number_input("Upper Case Ratio", min_value=0.0, max_value=1.0,
                                               value=sum(1 for c in review_text if c.isupper()) / len(review_text))
            title_text_length = st.number_input("Title Length", min_value=0, value=len(review_title))
            title_word_count = st.number_input("Title Word Count", min_value=0,
                                               value=len(review_title.split()))

        submitted = st.form_submit_button("Predict Rating")

    if submitted:
        # Prepare inputs
        num_input = np.array([[
            review_count,
            0,  # Placeholder for UserCountry_encoded (would need encoding logic)
            text_length,
            word_count,
            len(review_text) / max(1, word_count),  # avg_word_length
            exclamation_count,
            question_count,
            upper_case_ratio,
            title_text_length,
            title_word_count,
            len(review_title) / max(1, len(review_title.split())),  # title_avg_word_length
            review_title.count('!'),  # title_exclamation_count
            review_title.count('?'),  # title_question_count
            sum(1 for c in review_title if c.isupper()) / max(1, len(review_title))
        ]], dtype=np.float32)

        inputs = preprocess_inputs(model_name, review_text, num_input)

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


if __name__ == "__main__":
    main()