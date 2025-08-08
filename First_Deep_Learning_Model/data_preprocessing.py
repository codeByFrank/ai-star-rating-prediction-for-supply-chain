#!/usr/bin/env python3
"""
Data Preprocessing and Feature Engineering for Customer Satisfaction Prediction
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string
import warnings

warnings.filterwarnings('ignore')
# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class DataPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def clean_text(self, text):
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
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens
                  if token not in self.stop_words and len(token) > 2]

        return ' '.join(tokens)

    def extract_text_features(self, text):
        """Extract additional features from text"""
        if pd.isna(text) or text == '':
            return {
                'text_length': 0,
                'word_count': 0,
                'avg_word_length': 0,
                'exclamation_count': 0,
                'question_count': 0,
                'upper_case_ratio': 0
            }

        text = str(text)
        words = text.split()

        return {
            'text_length': len(text),
            'word_count': len(words),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'upper_case_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
        }

    def load_and_explore_data(self, filepath):
        """Load data and perform initial exploration"""
        print("Loading and exploring data...")
        df = pd.read_csv(filepath)

        print(f"Dataset shape: {df.shape}")
        print(f"\nTarget variable distribution:")
        print(df['ReviewRating'].value_counts().sort_index())

        # Calculate class imbalance ratio
        rating_counts = df['ReviewRating'].value_counts()
        imbalance_ratio = rating_counts.max() / rating_counts.min()
        print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}")

        # Visualize target distribution
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        df['ReviewRating'].value_counts().sort_index().plot(kind='bar')
        plt.title('Review Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Count')

        plt.subplot(1, 2, 2)
        df['ReviewRating'].value_counts().sort_index().plot(kind='pie', autopct='%1.1f%%')
        plt.title('Review Rating Distribution (%)')
        plt.ylabel('')

        plt.tight_layout()
        plt.savefig('charts/target_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        return df

    def preprocess_features(self, df):
        """Preprocess all features"""
        print("\nPreprocessing features...")

        # Clean text features
        print("Cleaning text data...")
        df['ReviewText_clean'] = df['ReviewText'].apply(self.clean_text)
        df['ReviewTitle_clean'] = df['ReviewTitle'].apply(self.clean_text)

        # Combine text features
        df['combined_text'] = df['ReviewText_clean'] + ' ' + df['ReviewTitle_clean']

        # Extract text features
        print("Extracting text features...")
        text_features = df['ReviewText'].apply(self.extract_text_features)
        text_features_df = pd.DataFrame(text_features.tolist())

        title_features = df['ReviewTitle'].apply(self.extract_text_features)
        title_features_df = pd.DataFrame(title_features.tolist())
        title_features_df.columns = ['title_' + col for col in title_features_df.columns]

        # Combine all features
        df = pd.concat([df, text_features_df, title_features_df], axis=1)

        # Encode categorical features
        categorical_features = ['UserCountry']
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[f'{feature}_encoded'] = le.fit_transform(df[feature].astype(str))
                self.label_encoders[feature] = le

        # Select final features for modeling
        feature_columns = [
            'ReviewCount', 'UserCountry_encoded',
            'text_length', 'word_count', 'avg_word_length',
            'exclamation_count', 'question_count', 'upper_case_ratio',
            'title_text_length', 'title_word_count', 'title_avg_word_length',
            'title_exclamation_count', 'title_question_count', 'title_upper_case_ratio'
        ]

        # Handle missing values
        for col in feature_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        return df, feature_columns

    def create_balanced_dataset(self, X, y, strategy='combined'):
        """Handle class imbalance using various techniques"""
        print(f"\nHandling class imbalance using {strategy} strategy...")

        if strategy == 'smote':
            # Use SMOTE for oversampling
            smote = SMOTE(random_state=42, k_neighbors=2)  # Reduced k_neighbors due to small dataset
            X_balanced, y_balanced = smote.fit_resample(X, y)

        elif strategy == 'undersampling':
            # Use random undersampling
            undersampler = RandomUnderSampler(random_state=42)
            X_balanced, y_balanced = undersampler.fit_resample(X, y)

        elif strategy == 'combined':
            # Combined approach: first oversample minority classes, then undersample majority
            # Step 1: Oversample very minority classes
            smote = SMOTE(random_state=42, k_neighbors=1)
            X_temp, y_temp = smote.fit_resample(X, y)

            # Step 2: Undersample majority class
            undersampler = RandomUnderSampler(random_state=42,
                                              sampling_strategy={1: 40})  # Reduce class 1 to 40 samples
            X_balanced, y_balanced = undersampler.fit_resample(X_temp, y_temp)

        else:
            X_balanced, y_balanced = X, y

        print(f"Original distribution: {np.bincount(y)}")
        print(f"Balanced distribution: {np.bincount(y_balanced)}")

        return X_balanced, y_balanced

    def prepare_text_sequences(self, df, max_features=5000, max_len=100):
        """Prepare text data for deep learning models"""
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        print("\nPreparing text sequences for deep learning...")

        # Initialize tokenizer
        tokenizer = Tokenizer(num_words=max_features, oov_token='<OOV>')
        tokenizer.fit_on_texts(df['combined_text'])

        # Convert texts to sequences
        sequences = tokenizer.texts_to_sequences(df['combined_text'])

        # Pad sequences
        X_text = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

        print(f"Text sequences shape: {X_text.shape}")
        print(f"Vocabulary size: {len(tokenizer.word_index)}")

        return X_text, tokenizer

    def split_data(self, X_numerical, X_text, y, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets"""
        print(
            f"\nSplitting data into train ({1 - test_size - val_size:.1%}), validation ({val_size:.1%}), and test ({test_size:.1%}) sets...")

        # First split: separate test set
        X_num_temp, X_num_test, X_text_temp, X_text_test, y_temp, y_test = train_test_split(
            X_numerical, X_text, y, test_size=test_size, random_state=42, stratify=y
        )

        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size
        X_num_train, X_num_val, X_text_train, X_text_val, y_train, y_val = train_test_split(
            X_num_temp, X_text_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )

        print(f"Train set size: {len(X_num_train)}")
        print(f"Validation set size: {len(X_num_val)}")
        print(f"Test set size: {len(X_num_test)}")

        # Scale numerical features
        X_num_train_scaled = self.scaler.fit_transform(X_num_train)
        X_num_val_scaled = self.scaler.transform(X_num_val)
        X_num_test_scaled = self.scaler.transform(X_num_test)

        return (X_num_train_scaled, X_num_val_scaled, X_num_test_scaled,
                X_text_train, X_text_val, X_text_test,
                y_train, y_val, y_test)


def main():
    # Create output directories
    import os
    os.makedirs('charts', exist_ok=True)

    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Load and explore data
    df = preprocessor.load_and_explore_data('data/temu_reviews_cleaned.csv')

    # Preprocess features
    df_processed, feature_columns = preprocessor.preprocess_features(df)

    # Prepare numerical features
    X_numerical = df_processed[feature_columns].values

    # Prepare text sequences
    X_text, tokenizer = preprocessor.prepare_text_sequences(df_processed)

    # Prepare target variable
    y = df_processed['ReviewRating'].values - 1  # Convert to 0-4 for neural networks

    # Split data
    (X_num_train, X_num_val, X_num_test,
     X_text_train, X_text_val, X_text_test,
     y_train, y_val, y_test) = preprocessor.split_data(X_numerical, X_text, y)

    # Handle class imbalance for training data only
    # Combine numerical and text features for balancing
    X_train_combined = np.concatenate([X_num_train, X_text_train], axis=1)
    X_train_balanced, y_train_balanced = preprocessor.create_balanced_dataset(
        X_train_combined, y_train, strategy='combined'
    )

    # Split back into numerical and text features
    num_features = X_num_train.shape[1]
    X_num_train_balanced = X_train_balanced[:, :num_features]
    X_text_train_balanced = X_train_balanced[:, num_features:].astype(int)

    # Calculate class weights for models
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    # Save preprocessed data
    np.savez('data/preprocessed_data.npz',
             X_num_train=X_num_train, X_num_val=X_num_val, X_num_test=X_num_test,
             X_text_train=X_text_train, X_text_val=X_text_val, X_text_test=X_text_test,
             X_num_train_balanced=X_num_train_balanced,
             X_text_train_balanced=X_text_train_balanced,
             y_train=y_train, y_val=y_val, y_test=y_test,
             y_train_balanced=y_train_balanced)

    # Save metadata
    metadata = {
        'feature_columns': feature_columns,
        'vocab_size': len(tokenizer.word_index) + 1,
        'max_sequence_length': X_text.shape[1],
        'num_classes': len(np.unique(y)),
        'class_weights': class_weight_dict,
        'class_names': ['Rating 1', 'Rating 2', 'Rating 3', 'Rating 4', 'Rating 5']
    }

    import pickle
    with open('data/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    with open('data/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    print("\n" + "=" * 50)
    print("Data preprocessing completed successfully!")
    print("=" * 50)
    print(f"Numerical features shape: {X_num_train.shape}")
    print(f"Text features shape: {X_text_train.shape}")
    print(f"Balanced training set size: {len(X_num_train_balanced)}")
    print(f"Class weights: {class_weight_dict}")

    return metadata


if __name__ == "__main__":
    main()