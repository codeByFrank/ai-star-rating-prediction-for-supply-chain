#!/usr/bin/env python3
"""
Deep Learning Models for Customer Satisfaction Prediction
Implements and compares 5 different deep learning architectures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, recall_score, precision_score
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential

from nltk.corpus import wordnet
from tensorflow.keras.layers import (
    Dense, LSTM, Bidirectional, Conv1D, MaxPooling1D, GlobalMaxPooling1D,
    Embedding, Dropout, Input, concatenate, Attention, MultiHeadAttention,
    LayerNormalization, Add, GlobalAveragePooling1D, BatchNormalization, SpatialDropout1D
)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    TerminateOnNaN,
    ModelCheckpoint,
    CSVLogger
)

from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
import pickle
import warnings
import os
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class DeepLearningModels:
    def __init__(self, vocab_size, max_len, num_features, num_classes, embedding_dim=128):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_features = num_features
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.models = {}
        self.histories = {}

    def create_mlp_model(self, input_dim):
        """Model 6: Deep MLP with TF-IDF features"""
        model = Sequential([ Input(shape=(input_dim,)),
                  # First hidden layer with batch normalization
                  Dense(1024, activation='relu', kernel_regularizer=l2(0.001), input_shape=(input_dim,)),
                  BatchNormalization(),
                  Dropout(0.6),
                  # Second hidden layer
                  Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
                  BatchNormalization(),
                  Dropout(0.5),
                  # Third hidden layer
                  Dense(256, activation='relu'),
                  Dropout(0.4),
                  Dense(128, activation='relu'),
                  # Output layer
                  Dense(self.num_classes, activation='softmax')
        ])
        return model


    def create_lstm_model(self):
        """Model 1: LSTM-based RNN for sequential text processing"""
        # Text input branch
        text_input = Input(shape=(self.max_len,))
        x = Embedding(self.vocab_size, self.embedding_dim, mask_zero=True)(text_input)
        x = SpatialDropout1D(0.3)(x)
        x = LSTM(128, return_sequences=True)(x)
        x = LSTM(64)(x)

        # Numerical input branch
        num_input = Input(shape=(self.num_features,))
        y = Dense(64, activation='relu')(num_input)
        y = BatchNormalization()(y)

        # Combine branches
        z = concatenate([x, y])
        z = Dense(128, activation='relu')(z)
        z = Dropout(0.5)(z)
        output = Dense(self.num_classes, activation='softmax')(z)

        model = Model(inputs=[text_input, num_input], outputs=output)
        optimizer = Adam(learning_rate=0.001, clipvalue=0.5)
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def create_bilstm_attention_model(self):
        """Model 2: Bidirectional LSTM with attention mechanism"""
        # Text input branch
        text_input = Input(shape=(self.max_len,))
        x = Embedding(self.vocab_size, self.embedding_dim, mask_zero=True)(text_input)
        x = Bidirectional(LSTM(128, return_sequences=True))(x)

        # Attention
        attention = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = LayerNormalization()(x + attention)

        # Numerical branch
        num_input = Input(shape=(self.num_features,))
        y = Dense(64, activation='swish')(num_input)
        y = LayerNormalization()(y)

        # Combined
        context = GlobalAveragePooling1D()(x)
        z = concatenate([context, y])
        z = Dense(256, activation='swish')(z)
        z = Dropout(0.4)(z)
        output = Dense(self.num_classes, activation='softmax')(z)

        model = Model(inputs=[text_input, num_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.0005),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def create_robust_model(self):
        # Text
        text_input = Input(shape=(self.max_len,))
        x = Embedding(self.vocab_size, 128)(text_input)
        x = Bidirectional(LSTM(64))(x)

        # Numerical
        num_input = Input(shape=(self.num_features,))
        y = Dense(64)(num_input)

        # Combined
        z = concatenate([x, y])
        z = Dense(128, activation='relu')(z)
        output = Dense(self.num_classes, activation='softmax')(z)

        model = Model(inputs=[text_input, num_input], outputs=output)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def create_accurate_bilstm_attention_model(self):
        """High-performance BiLSTM with Attention"""
        # Text input branch
        text_input = Input(shape=(self.max_len,), name='text_input')
        text_embedding = Embedding(
            self.vocab_size,
            self.embedding_dim * 2,  # Increased capacity
            mask_zero=True
        )(text_input)

        # Enhanced Bidirectional LSTM
        bilstm = Bidirectional(
            LSTM(128,  # Doubled units
                 dropout=0.3,
                 recurrent_dropout=0.25,
                 return_sequences=True,
                 kernel_regularizer=l2(1e-4))  # Added regularization
        )(text_embedding)
        bilstm = BatchNormalization()(bilstm)  # Stabilizes training

        # Powerful attention mechanism
        attention = MultiHeadAttention(
            num_heads=8,  # More attention heads
            key_dim=128,  # Matches LSTM units
            dropout=0.2,
            kernel_regularizer=l2(1e-4)
        )(bilstm, bilstm)

        # Residual connection with layer norm
        attention = Add()([bilstm, attention])
        attention = LayerNormalization()(attention)

        # Context extraction
        text_features = GlobalAveragePooling1D()(attention)

        # Enhanced numerical branch
        num_input = Input(shape=(self.num_features,), name='numerical_input')
        num_dense = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(num_input)
        num_dense = BatchNormalization()(num_dense)

        # Feature fusion
        combined = concatenate([text_features, num_dense])
        combined = Dropout(0.4)(combined)

        # Deep classifier head
        hidden = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(combined)
        hidden = BatchNormalization()(hidden)
        hidden = Dropout(0.4)(hidden)
        hidden = Dense(128, activation='relu')(hidden)
        output = Dense(self.num_classes, activation='softmax')(hidden)

        model = Model(inputs=[text_input, num_input], outputs=output)
        return model

    def create_cnn_model(self):
        """Model 3: CNN for text classification with multiple filter sizes"""
        # Text input branch
        text_input = Input(shape=(self.max_len,))
        x = Embedding(self.vocab_size, 128)(text_input)

        # Parallel conv branches with residual connections
        convs = []
        for filter_size in [3, 5, 7]:
            conv = Conv1D(128, filter_size, padding='same', activation='relu')(x)
            conv = MaxPooling1D(2)(conv)
            conv = Conv1D(64, filter_size, padding='same', activation='relu')(conv)
            conv = GlobalMaxPooling1D()(conv)
            convs.append(conv)

        x = concatenate(convs) if len(convs) > 1 else convs[0]

        # Numerical branch
        num_input = Input(shape=(self.num_features,))
        y = Dense(64, activation='relu')(num_input)
        y = BatchNormalization()(y)

        # Combined
        z = concatenate([x, y])
        z = Dense(128, activation='relu')(z)
        z = Dropout(0.5)(z)
        output = Dense(self.num_classes, activation='softmax')(z)

        model = Model(inputs=[text_input, num_input], outputs=output)
        model.compile(optimizer=RMSprop(learning_rate=0.0005),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def create_transformer_model(self):
        """Model 4: Transformer-based model (simplified BERT-like architecture)"""
        # Text input branch
        text_input = Input(shape=(self.max_len,))
        x = Embedding(self.vocab_size, 128)(text_input)

        # Positional encoding
        positions = tf.range(start=0, limit=self.max_len, delta=1)
        positions = Embedding(self.max_len, 128)(positions)
        x = x + positions

        # Transformer blocks
        for _ in range(3):  # Additional layer
            attn = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
            x = LayerNormalization()(x + attn)
            ffn = Dense(512, activation='gelu')(x)
            ffn = Dense(128)(ffn)
            x = LayerNormalization()(x + ffn)

        x = GlobalAveragePooling1D()(x)

        # Numerical branch
        num_input = Input(shape=(self.num_features,))
        y = Dense(64, activation='relu')(num_input)
        y = LayerNormalization()(y)

        # Combined
        z = concatenate([x, y])
        z = Dense(256, activation='gelu')(z)
        z = Dropout(0.4)(z)
        output = Dense(self.num_classes, activation='softmax')(z)

        model = Model(inputs=[text_input, num_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.0001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def create_hybrid_cnn_lstm_model(self):
        """Model 5: Hybrid CNN-LSTM model"""
        # Text input branch
        text_input = Input(shape=(self.max_len,))
        x = Embedding(self.vocab_size, 128)(text_input)

        # CNN part
        conv1 = Conv1D(128, 3, padding='same', activation='relu')(x)
        conv1 = MaxPooling1D(2)(conv1)
        conv2 = Conv1D(128, 5, padding='same', activation='relu')(x)
        conv2 = MaxPooling1D(2)(conv2)
        x = concatenate([conv1, conv2])
        x = BatchNormalization()(x)

        # LSTM part
        x = Bidirectional(LSTM(128))(x)

        # Numerical branch
        num_input = Input(shape=(self.num_features,))
        y = Dense(64, activation='relu')(num_input)
        y = BatchNormalization()(y)

        # Combined
        z = concatenate([x, y])
        z = Dense(256, activation='relu')(z)
        z = Dropout(0.5)(z)
        output = Dense(self.num_classes, activation='softmax')(z)

        model = Model(inputs=[text_input, num_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.0005),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def create_hybrid_cnn_lstm_model_modified(self):
        """Fixed version with all required imports"""
        # Text input branch
        text_input = Input(shape=(self.max_len,), name='text_input')
        text_embedding = Embedding(self.vocab_size, self.embedding_dim)(text_input)

        # CNN with MaxPooling
        conv1 = Conv1D(128, 3, activation='relu', padding='same')(text_embedding)
        conv1 = MaxPooling1D(2)(conv1)
        conv2 = Conv1D(128, 5, activation='relu', padding='same')(text_embedding)
        conv2 = MaxPooling1D(2)(conv2)

        conv_combined = concatenate([conv1, conv2])
        conv_combined = BatchNormalization()(conv_combined)  # Now properly imported

        # LSTM
        lstm_out = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(conv_combined)

        # Numerical branch
        num_input = Input(shape=(self.num_features,), name='numerical_input')
        num_dense = Dense(32, activation='relu')(num_input)

        # Combine branches
        combined = concatenate([lstm_out, num_dense])
        hidden = Dense(128, activation='relu')(combined)
        output = Dense(self.num_classes, activation='softmax')(hidden)

        model = Model(inputs=[text_input, num_input], outputs=output)
        return model

    def compile_model(self, model, learning_rate=0.001):
        """Compile model with appropriate optimizer and loss function"""
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def create_callbacks(self):
        """Create training callbacks"""
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7
        )

        return [early_stopping, reduce_lr]

    def train_model_mlp(self, model, model_name, X_train, X_val, y_train, y_val,
                    class_weights=None, epochs=100, batch_size=128):
        """Train a model with given data"""
        print(f"\nTraining {model_name}...")

        #callbacks = self.create_callbacks()
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-5),
            TerminateOnNaN()
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )

        self.models[model_name] = model
        self.histories[model_name] = history

        return model, history

    def train_model(self, model, model_name, X_text_train, X_num_train, y_train,
                    X_text_val, X_num_val, y_val, class_weights, epochs=100):
        """Train a model with given data"""
        print(f"\nTraining {model_name}...")

        callbacks = self.create_callbacks()

        history = model.fit(
            [X_text_train, X_num_train], y_train,
            validation_data=([X_text_val, X_num_val], y_val),
            epochs=epochs,
            batch_size=16,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )

        self.models[model_name] = model
        self.histories[model_name] = history

        return model, history

    def evaluate_model(self, model, model_name, X_text_test, X_num_test, y_test, class_names):
        """Evaluate model performance"""
        print(f"\nEvaluating {model_name}...")

        # Make predictions - handle MLP vs other models differently
        if 'MLP' in model_name:
            # MLP expects single input array
            y_pred_proba = model.predict(X_text_test)  # X_text_test actually contains all features for MLP
        else:
            # Other models expect separate text and numerical inputs
            y_pred_proba = model.predict([X_text_test, X_num_test])

        # Make predictions
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        micro_precision = precision_score(y_test, y_pred, average='micro')
        micro_recall = recall_score(y_test, y_pred, average='micro')

        # Multi-class ROC AUC
        try:
            auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            auc_score = 0.0

        # Precision, Recall, F1 per class
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)

        # Classification report
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'auc_score': auc_score,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'precision': precision,
            'recall': recall,
            'f1_per_class': f1,
            'support': support,
            'classification_report': report,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

        return results

    def plot_training_history(self, model_name):
        """Plot training history"""
        if model_name not in self.histories:
            return

        history = self.histories[model_name]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title(f'{model_name} - Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title(f'{model_name} - Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f'charts/{model_name.lower().replace(" ", "_")}_training_history.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrix(self, results, class_names):
        """Plot confusion matrix"""
        cm = results['confusion_matrix']
        model_name = results['model_name']

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{model_name} - Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'charts/{model_name.lower().replace(" ", "_")}_confusion_matrix.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def save_all_models(self, all_results):
        """Save all trained models with their evaluation results and supporting files for API use"""
        # Create directories if they don't exist
        os.makedirs('api_models', exist_ok=True)
        os.makedirs('api_models/data', exist_ok=True)

        # Actual features from your Temu reviews dataset
        feature_columns = [
            'ReviewCount', 'UserCountry_encoded',
            'text_length', 'word_count', 'avg_word_length',
            'exclamation_count', 'question_count', 'upper_case_ratio',
            'title_text_length', 'title_word_count', 'title_avg_word_length',
            'title_exclamation_count', 'title_question_count', 'title_upper_case_ratio'
        ]

        # Class names based on ReviewRating (1-5 stars)
        class_names = [
            '1 Star - Very Poor',
            '2 Stars - Poor',
            '3 Stars - Average',
            '4 Stars - Good',
            '5 Stars - Excellent'
        ]

        # Create a package for each model that contains everything needed for serving
        for result in all_results:
            model_name = result['model_name']
            if model_name in self.models:
                # Create a directory for this model
                model_dir = os.path.join('api_models', model_name.lower().replace(' ', '_'))
                os.makedirs(model_dir, exist_ok=True)

                # 1. Save the model in SavedModel format
                model_path = os.path.join(model_dir, 'model.keras')
                self.models[model_name].save(model_path)

                # 2. Save metadata needed for preprocessing
                metadata = {
                    'max_sequence_length': self.max_len,
                    'feature_columns': feature_columns,
                    'class_names': class_names,
                    'input_details': {
                        'text_input': {
                            'shape': [None, self.max_len],
                            'dtype': 'int32',
                            'description': 'Tokenized review text from ReviewText column'
                        },
                        'numerical_input': {
                            'shape': [None, len(feature_columns)],
                            'dtype': 'float32',
                            'description': f'Numerical features in order: {", ".join(feature_columns)}'
                        }
                    },
                    'output_details': {
                        'description': 'Probability scores for each rating level (1-5 stars)',
                        'class_order': class_names
                    },
                    'data_source': 'temu_reviews_cleaned.csv',
                    'text_columns_used': ['ReviewText', 'ReviewTitle'],  # Which text columns were used
                    'model_format': 'keras'  # Indicate the saved format
                }

                with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
                    json.dump(metadata, f, indent=2)

                print(f"✅ Saved API-ready {model_name} package to {model_dir}")

                # Update the result with the path
                result['api_model_path'] = model_dir
            else:
                print(f"⚠️ Model {model_name} not found in trained models")

        # Save tokenizer if exists
        if hasattr(self, 'tokenizer'):
            tokenizer_path = os.path.join('api_models', 'tokenizer.pkl')
            with open(tokenizer_path, 'wb') as f:
                pickle.dump(self.tokenizer, f)
            print(f"✅ Saved tokenizer to {tokenizer_path}")

        # Save complete results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join('api_models', 'data', f'model_results_{timestamp}.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(all_results, f)

        print(f"\nAll models saved in API-ready format.")
        print(f"You can now deploy any model by copying its directory to your API server.")
        return results_path


def main():
    # Load preprocessed data
    print("Loading preprocessed data...")
    data = np.load('data/preprocessed_data.npz')
    
    X_num_train = data['X_num_train']
    X_num_val = data['X_num_val']
    X_num_test = data['X_num_test']
    X_text_train = data['X_text_train']
    X_text_val = data['X_text_val']
    X_text_test = data['X_text_test']
    X_num_train_balanced = data['X_num_train_balanced']
    X_text_train_balanced = data['X_text_train_balanced']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    y_train_balanced = data['y_train_balanced']

    # Load TF-IDF based data
    X_train_mlp = data['X_train_mlp']
    X_val_mlp = data['X_val_mlp']
    X_test_mlp = data['X_test_mlp']
    y_train_mlp = data['y_train_mlp']
    y_val_mlp = data['y_val_mlp']
    y_test_mlp = data['y_test_mlp']
    
    # Load metadata
    with open('data/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    vocab_size = metadata['vocab_size']
    max_len = metadata['max_sequence_length']
    num_features = len(metadata['feature_columns'])
    num_classes = metadata['num_classes']
    class_weights = metadata['class_weights']
    class_names = metadata['class_names']
    
    print(f"Vocab size: {vocab_size}")
    print(f"Max sequence length: {max_len}")
    print(f"Number of features: {num_features}")
    print(f"Number of classes: {num_classes}")
    print(f"Training with balanced data: {X_text_train_balanced.shape[0]} samples")
    
    # Initialize model builder
    model_builder = DeepLearningModels(vocab_size, max_len, num_features, num_classes, embedding_dim=128)
    
    # Define models to train
    model_configs = [
        ('Deep MLP with TF-IDF', lambda: model_builder.create_mlp_model(X_train_mlp.shape[1])),
        ('LSTM Model', model_builder.create_lstm_model),
        ('BiLSTM with Attention', model_builder.create_robust_model),
        ('CNN Model', model_builder.create_cnn_model),
        ('Transformer Model', model_builder.create_transformer_model),
        ('Hybrid CNN-LSTM', model_builder.create_hybrid_cnn_lstm_model)
    ]

    #model_configs = [
    #    ('CNN Model', model_builder.create_cnn_model)
    #]
    
    # Train and evaluate all models
    all_results = []
    
    for model_name, model_func in model_configs:
        print(f"\n{'='*50}")
        print(f"Building and training {model_name}")
        print('='*50)
        
        # Create and compile model
        model = model_func()
        model = model_builder.compile_model(model)
        
        print(f"\n{model_name} Architecture:")
        model.summary()

        # Special handling for MLP model (uses different data)
        if 'MLP' in model_name:
            # Handle class imbalance for MLP data
            smote = SMOTE(random_state=42)
            X_train_mlp_balanced, y_train_mlp_balanced = smote.fit_resample(X_train_mlp, y_train_mlp)

            # Train MLP model
            model, history = model_builder.train_model_mlp(
                model, model_name,
                X_train_mlp_balanced, X_val_mlp, y_train_mlp_balanced, y_val_mlp,
                class_weights, epochs=50, batch_size=64
            )

            # Evaluate MLP model
            results = model_builder.evaluate_model(
                model, model_name, X_test_mlp, None, y_test_mlp, class_names
            )
        else:
            # Train sequence-based models
            model, history = model_builder.train_model(
                model, model_name,
                X_text_train_balanced, X_num_train_balanced, y_train_balanced,
                X_text_val, X_num_val, y_val,
                class_weights, epochs=50
            )
            # Evaluate model
            results = model_builder.evaluate_model(
                model, model_name, X_text_test, X_num_test, y_test, class_names
            )

        # Plot training history
        model_builder.plot_training_history(model_name)

        # Plot confusion matrix
        model_builder.plot_confusion_matrix(results, class_names)
        
        all_results.append(results)
        
        # Print results
        print(f"\n{model_name} Results:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1-Score (Weighted): {results['f1_weighted']:.4f}")
        print(f"F1-Score (Macro): {results['f1_macro']:.4f}")
        print(f"AUC Score: {results['auc_score']:.4f}")
        print(results.keys())  # Check available keys
        print(f"Micro-averaged precision: {results['micro_precision']:.4f}")
        print(f"Micro-average recall: {results['micro_recall']:.4f}")
        print("Precision:", ", ".join([f"{x:.4f}" for x in results['precision']]))
        print("Recall:", ", ".join([f"{x:.4f}" for x in results['recall']]))

    # Save results
    with open('data/model_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    # Create comparison summary
    summary_data = []
    for result in all_results:
        summary_data.append({
            'Model': result['model_name'],
            'Accuracy': result['accuracy'],
            'F1-Weighted': result['f1_weighted'],
            'F1-Macro': result['f1_macro'],
            'AUC Score': result['auc_score'],
            'Micro-averaged precision': result['micro_precision'],
            'Micro-averaged recal': result['micro_recall']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('F1-Weighted', ascending=False)
    
    print(f"\n{'='*80}")
    print("MODEL PERFORMANCE COMPARISON")
    print('='*80)
    print(summary_df.to_string(index=False, float_format='%.4f'))
    
    # Save summary
    summary_df.to_csv('data/model_comparison_summary.csv', index=False)
    
    # Plot comparison
    plt.figure(figsize=(14, 8))
    
    metrics = ['Accuracy', 'F1-Weighted', 'F1-Macro', 'AUC Score', 'Micro-averaged precision', 'Micro-averaged recal']
    x = np.arange(len(summary_df))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, summary_df[metric], width, label=metric, alpha=0.8)
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Deep Learning Models Performance Comparison')
    plt.xticks(x + width*1.5, summary_df['Model'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('charts/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nBest performing model: {summary_df.iloc[0]['Model']}")
    print(f"Best F1-Weighted Score: {summary_df.iloc[0]['F1-Weighted']:.4f}")

    # After training all models, save everything
    results_path = model_builder.save_all_models(all_results)
    print(f"All models and results saved. Results path: {results_path}")
    
    return all_results, summary_df

if __name__ == "__main__":
    results, summary = main()
