from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import os
import json
from typing import Dict, List

app = FastAPI()

# Model configuration
MODEL_DIRS = {
    "lstm": "api_models/lstm_model",
    "bilstm": "api_models/bilstm_with_attention",
    "cnn": "api_models/cnn_model",
    "transformer": "api_models/transformer_model",
    "hybrid": "api_models/hybrid_cnn-lstm"
}

# Load all models and metadata
models = {}
for name, path in MODEL_DIRS.items():
    try:
        with open(os.path.join(path, "metadata.json")) as f:
            metadata = json.load(f)

        model = tf.keras.models.load_model(os.path.join(path, "model.keras"))

        models[name] = {
            "model": model,
            "metadata": metadata,
            "tokenizer": None
        }

        # Load tokenizer if exists
        tokenizer_path = os.path.join(path, "tokenizer.pkl")
        if os.path.exists(tokenizer_path):
            import pickle

            with open(tokenizer_path, "rb") as f:
                models[name]["tokenizer"] = pickle.load(f)

        print(f"✅ Loaded {name} model successfully")

    except Exception as e:
        print(f"⚠️ Failed to load {name} model: {str(e)}")


class NumericalFeatures(BaseModel):
    ReviewCount: int = 1
    UserCountry_encoded: int = 0


class PredictionRequest(BaseModel):
    review_text: str
    review_title: str = ""
    numerical_features: NumericalFeatures = NumericalFeatures()


def prepare_features(request: PredictionRequest, max_len: int, tokenizer=None):
    """Prepare input features for prediction"""
    # Text processing
    combined_text = f"{request.review_title}. {request.review_text}"

    if tokenizer:
        text_sequence = tokenizer.texts_to_sequences([combined_text])
        text_input = tf.keras.preprocessing.sequence.pad_sequences(
            text_sequence, maxlen=max_len
        )
    else:
        text_input = np.zeros((1, max_len))

    # Numerical features (14 total)
    num_input = np.zeros((1, 14))
    num_input[0, 0] = request.numerical_features.ReviewCount
    num_input[0, 1] = request.numerical_features.UserCountry_encoded

    # Auto-calculate text features (positions 2-13)
    def calc_text_features(text):
        if not text: return [0] * 6
        words = text.split()
        upper = sum(1 for c in text if c.isupper())
        return [
            len(text),
            len(words),
            sum(len(w) for w in words) / len(words) if words else 0,
            text.count('!'),
            text.count('?'),
            upper / max(1, len(text))
        ]

    # Review text features (positions 2-7)
    num_input[0, 2:8] = calc_text_features(request.review_text)
    # Title features (positions 8-13)
    num_input[0, 8:14] = calc_text_features(request.review_title)

    return text_input, num_input


@app.post("/predict")
async def predict_all_models(request: PredictionRequest):
    results = {}

    for model_name, model_data in models.items():
        try:
            # Prepare inputs
            text_input, num_input = prepare_features(
                request,
                max_len=model_data["metadata"]["max_sequence_length"],
                tokenizer=model_data["tokenizer"]
            )

            # Make prediction
            pred = model_data["model"].predict({
                'text_input': text_input,
                'numerical_input': num_input
            })

            # Get predicted class
            class_idx = np.argmax(pred[0])
            results[model_name] = {
                "predicted_class": model_data["metadata"]["class_names"][class_idx],
                "confidence": float(pred[0][class_idx]),
                "all_predictions": {
                    cls: float(prob) for cls, prob in
                    zip(model_data["metadata"]["class_names"], pred[0])
                }
            }

        except Exception as e:
            results[model_name] = {"error": str(e)}

    return {
        "input_text": request.review_text,
        "predictions": results
    }


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