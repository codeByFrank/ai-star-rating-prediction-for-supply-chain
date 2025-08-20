from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import os
import json
import pickle
from typing import Dict, List
from scipy.sparse import hstack  # For feature padding


app = FastAPI()

# Model configuration
MODEL_DIRS = {
    "lstm": "api_models/lstm_model",
    "bilstm": "api_models/bilstm_with_attention",
    "cnn": "api_models/cnn_model",
    "transformer": "api_models/transformer_model",
    "hybrid": "api_models/hybrid_cnn-lstm",
    "deep_mlp": "api_models/deep_mlp_with_tf-idf"
}

# Load all models and metadata
models = {}
for name, path in MODEL_DIRS.items():
    try:
        with open(os.path.join(path, "metadata.json")) as f:
            metadata = json.load(f)

        model = tf.keras.models.load_model(os.path.join(path, "model.keras"))

        # Special handling for deep_mlp (TF-IDF model)
        if name == "deep_mlp":
            # Load TF-IDF vectorizer
            vectorizer_path = os.path.join(path, "tfidf.pkl")
            with open(vectorizer_path, "rb") as f:
                vectorizer = pickle.load(f)
                # Verify feature count matches model expectations
                expected_features = 10014  # From error message
                vocab_size = len(vectorizer.vocabulary_)
                if vocab_size != expected_features:
                    print(f"⚠️ Vectorizer has {vocab_size} features, expected {expected_features}")

            models[name] = {
                "model": model,
                "metadata": metadata,
                "vectorizer": vectorizer,
                "input_type": "tfidf",
                "expected_features": expected_features
            }
            print(f"✅ Loaded {name} model with TF-IDF vectorizer")
            continue

        # Standard model loading for other models
        input_names = [inp.name for inp in model.inputs]
        if len(input_names) != 2:
            raise ValueError(f"Model {name} expects {len(input_names)} inputs, expected 2")

        # Determine input types
        text_input_name = input_names[0] if "text" in input_names[0].lower() or \
                                            metadata["max_sequence_length"] in model.inputs[0].shape else \
            input_names[1]
        numerical_input_name = input_names[1] if input_names[0] == text_input_name else input_names[0]

        # Load tokenizer if exists
        tokenizer = None
        tokenizer_path = os.path.join(path, "tokenizer.pkl")
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, "rb") as f:
                tokenizer = pickle.load(f)

        models[name] = {
            "model": model,
            "metadata": metadata,
            "tokenizer": tokenizer,
            "input_map": {
                "text": text_input_name,
                "numerical": numerical_input_name
            }
        }
        print(f"✅ Loaded {name} model successfully")

    except Exception as e:
        print(f"⚠️ Failed to load {name} model: {str(e)}")
        models[name] = {"error": str(e)}


class NumericalFeatures(BaseModel):
    ReviewCount: int = 1
    UserCountry_encoded: int = 0


class PredictionRequest(BaseModel):
    review_text: str
    review_title: str = ""
    numerical_features: NumericalFeatures = NumericalFeatures()


def prepare_features(request: PredictionRequest, max_len: int, tokenizer=None):
    """Prepare input features for prediction"""
    try:
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

    except Exception as e:
        raise HTTPException(400, f"Input preparation failed: {str(e)}")

@app.post("/predict")
async def predict_all_models(request: PredictionRequest):
    results = {}

    for model_name, model_data in models.items():
        if "error" in model_data:
            results[model_name] = {"error": model_data["error"]}
            continue

        try:
            combined_text = f"{request.review_title}. {request.review_text}"

            if model_name == "deep_mlp":
                # TF-IDF model processing
                if "vectorizer" not in model_data:
                    raise ValueError("TF-IDF vectorizer not loaded")

                # Transform text to TF-IDF features
                tfidf_features = model_data["vectorizer"].transform([combined_text])

                # Ensure correct feature dimensions
                current_features = tfidf_features.shape[1]
                expected = model_data["expected_features"]

                if current_features < expected:
                    # Pad with zeros
                    missing = expected - current_features
                    tfidf_features = hstack([tfidf_features, np.zeros((1, missing))])
                elif current_features > expected:
                    # Truncate (shouldn't happen with proper vectorizer)
                    tfidf_features = tfidf_features[:, :expected]

                # Convert to dense array with correct shape
                input_features = tfidf_features.toarray()
                print(f"TF-IDF features shape: {input_features.shape}")  # Should be (1, 10014)

                # Predict
                pred = model_data["model"].predict(input_features)

            elif "input_map" in model_data:
                # Standard dual-input models
                text_input, num_input = prepare_features(
                    request,
                    max_len=model_data["metadata"]["max_sequence_length"],
                    tokenizer=model_data["tokenizer"]
                )

                model_inputs = {
                    model_data["input_map"]["text"]: text_input,
                    model_data["input_map"]["numerical"]: num_input
                }
                pred = model_data["model"].predict(model_inputs)
            else:
                raise ValueError("Unknown model type configuration")

            # Format results
            class_idx = np.argmax(pred[0])
            results[model_name] = {
                "predicted_class": model_data["metadata"]["class_names"][class_idx],
                "confidence": float(pred[0][class_idx]),
                "all_predictions": {
                    cls: float(prob) for cls, prob in
                    zip(model_data["metadata"]["class_names"], pred[0])
                },
                "model_type": model_name
            }

        except Exception as e:
            results[model_name] = {
                "error": f"{type(e).__name__}: {str(e)}",
                "model_type": model_name
            }

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