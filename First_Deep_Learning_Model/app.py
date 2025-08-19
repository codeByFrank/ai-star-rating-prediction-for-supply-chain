# api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import pickle
import json
import os
from typing import List

app = FastAPI()

# Model configuration
MODEL_DIR = "api_models/lstm_model"  # Change to your best performing model

# Load metadata
with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
    metadata = json.load(f)

print("Expected features:", metadata["feature_columns"])
model = tf.keras.models.load_model("api_models/lstm_model/model.keras")

# For numerical input (assuming it's the second input)
num_input_layer = model.inputs[1]
print("Expected numerical input shape:", num_input_layer.shape)  # Should show (None, 14)

# Load tokenizer if exists
tokenizer_path = os.path.join("api_models", "tokenizer.pkl")
if os.path.exists(tokenizer_path):
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

# Load model - now using .keras extension
model_path = os.path.join(MODEL_DIR, "model.keras")
model = tf.keras.models.load_model(model_path)


class NumericalFeatures(BaseModel):
    ReviewCount: int  # Number of reviews user has made
    #ReviewRating: float  # Rating given (1-5 stars)
    UserCountry_encoded: int = 0  # Default to 0 if not provided
    # Add any other numerical features you're using


class PredictionRequest(BaseModel):
    review_text: str  # The main review text (ReviewText column)
    review_title: str  # The review title (ReviewTitle column)
    #review_count: int
    numerical_features: NumericalFeatures


def prepare_features(review_text: str, review_title: str, review_count: int) -> np.ndarray:
    """Prepare all 14 features expected by the model"""
    import re

    # Initialize with zeros (will fill available features)
    features = np.zeros(14)

    # 1. ReviewCount (from input)
    features[0] = review_count

    # 2. UserCountry_encoded (you'll need to implement encoding)
    features[1] = encode_country(user_country)  # Add if you have country data

    # Text statistics for review body (features 2-8)
    features[2] = len(review_text)  # text_length
    features[3] = len(review_text.split())  # word_count
    features[4] = np.mean([len(word) for word in review_text.split()]) if review_text else 0  # avg_word_length
    features[5] = review_text.count('!')  # exclamation_count
    features[6] = review_text.count('?')  # question_count
    features[7] = sum(1 for c in review_text if c.isupper()) / len(
        review_text) if review_text else 0  # upper_case_ratio

    # Title statistics (features 9-14)
    features[8] = len(review_title)  # title_text_length
    features[9] = len(review_title.split())  # title_word_count
    features[10] = np.mean([len(word) for word in review_title.split()]) if review_title else 0  # title_avg_word_length
    features[11] = review_title.count('!')  # title_exclamation_count
    features[12] = review_title.count('?')  # title_question_count
    features[13] = sum(1 for c in review_title if c.isupper()) / len(
        review_title) if review_title else 0  # title_upper_case_ratio

    return features.reshape(1, -1)  # Reshape to (1, 14)


@app.post("/predict_review_rating")
async def predict_review_rating(request: PredictionRequest):
    try:
        # Prepare all 14 features
        num_input = np.zeros((1, 14))
        num_input = prepare_features(
            review_text=request.review_text,
            review_title=request.review_title,
            review_count=request.review_count
            # user_country=request.user_country  # Uncomment if available
        )
        num_input = np.zeros((1, 14))
        num_input[0, 0] = request.numerical_features.ReviewCount  # Position 0
        num_input[0, 1] = request.numerical_features.UserCountry_encoded  # Position 1

        # Auto-calculate the remaining 12 text features
        num_input[0, 2] = len(request.review_text)  # text_length
        num_input[0, 3] = len(request.review_text.split())  # word_count

        # Combine text inputs (you might want to adjust this based on your preprocessing)
        combined_text = f"{request.review_title}. {request.review_text}"
        text_sequence = tokenizer.texts_to_sequences([combined_text])
        text_input = tf.keras.preprocessing.sequence.pad_sequences(
            text_sequence,
            maxlen=metadata['max_sequence_length']
        )

        # Preprocess text input
        if 'tokenizer' in globals():
            text_sequence = tokenizer.texts_to_sequences([combined_text])
            text_input = tf.keras.preprocessing.sequence.pad_sequences(
                text_sequence,
                maxlen=metadata['max_sequence_length']
            )
        else:
            text_input = np.zeros((1, metadata['max_sequence_length']))

        # Prepare numerical input in the exact order of feature_columns
        num_input = np.array([[
            request.numerical_features.ReviewCount,
            request.numerical_features.ReviewRating,
            # Add any other numerical features here in the same order as feature_columns
        ]])
        # Debug shapes
        print("Text input shape:", text_input.shape)
        print("Numerical input shape:", num_input.shape)

        # Make prediction
        prediction = model.predict({
            'text_input': text_input,
            'numerical_input': num_input
        })

        # Format predictions with class names and scores
        predictions_with_classes = [
            {"rating": i + 1, "label": class_name, "score": float(score)}
            for i, (class_name, score) in enumerate(zip(metadata['class_names'], prediction[0]))
        ]

        # Get the predicted rating (1-5)
        predicted_rating = np.argmax(prediction) + 1
        predicted_class = metadata['class_names'][predicted_rating - 1]

        return {
            "predicted_rating": predicted_rating,
            "predicted_label": predicted_class,
            "confidence": float(np.max(prediction)),
            "all_predictions": predictions_with_classes,
            "input_features": {
                "text_length": len(combined_text),
                "numerical_features": {
                    "ReviewCount": int(request.numerical_features.ReviewCount),
                    "ReviewRating": float(request.numerical_features.ReviewRating)
                }
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # Use these default settings that work for most environments
    HOST = "127.0.0.1"  # Localhost - more reliable than "0.0.0.0" for development
    PORT = 8000  # Default FastAPI port

    # Check if port is available
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((HOST, PORT))
    except socket.error as e:
        print(f"Port {PORT} is already in use. Try a different port.")
        PORT = 8001  # Fallback port

    sock.close()

    print(f"Starting API server at http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)