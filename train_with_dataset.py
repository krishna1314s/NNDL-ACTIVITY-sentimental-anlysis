"""
Train Sentiment Analysis RNN with Real Dataset
Downloads and uses IMDB movie reviews dataset for training
Optimized for faster training
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split
import pickle
import os
import time

class FastSentimentRNN:
    def __init__(self, vocab_size=2000, embedding_dim=128, max_length=200, 
                 lstm_units=128, num_classes=1):
        """Initialize with optimized hyperparameters"""
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.lstm_units = lstm_units
        self.num_classes = num_classes
        self.tokenizer = None
        self.model = None
        
    def build_model(self):
        """Build the RNN model architecture"""
        self.model = Sequential([
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_length
            ),
            Bidirectional(
                LSTM(
                    units=self.lstm_units,
                    return_sequences=False,
                    dropout=0.2,
                    recurrent_dropout=0.2
                )
            ),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=10, batch_size=128, validation_split=0.2):
        """Train with optimized settings for speed"""
        if self.model is None:
            self.build_model()
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=2,  # Reduced patience for faster training
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'best_sentiment_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=0  # Less verbose for speed
            )
        ]
        
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = None
        else:
            validation_data = None
        
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,  # Larger batch size for speed
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, texts):
        """Predict sentiment for given texts"""
        if isinstance(texts, str):
            texts = [texts]
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(
            sequences,
            maxlen=self.max_length,
            padding='post',
            truncating='post'
        )
        
        predictions = self.model.predict(padded, verbose=0)
        return predictions.flatten()
    
    def predict_class(self, texts, threshold=0.5):
        """Predict sentiment class"""
        predictions = self.predict(texts)
        classes = (predictions >= threshold).astype(int)
        return classes
    
    def save_model(self, model_path='sentiment_rnn_model.h5', tokenizer_path='tokenizer.pkl'):
        """Save the trained model and tokenizer"""
        if self.model is not None:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
        
        if self.tokenizer is not None:
            with open(tokenizer_path, 'wb') as f:
                pickle.dump(self.tokenizer, f)
            print(f"Tokenizer saved to {tokenizer_path}")


def load_imdb_data(num_words=2000, max_length=200, test_size=0.2):
    """
    Load IMDB movie reviews dataset
    This is a built-in dataset with 50,000 reviews
    """
    print("Loading IMDB movie reviews dataset...")
    print("This may take a moment on first run (downloading dataset)...")
    
    # Load IMDB dataset (already tokenized)
    (X_train_full, y_train_full), (X_test, y_test) = imdb.load_data(
        num_words=num_words
    )
    
    print(f"Loaded {len(X_train_full)} training samples")
    print(f"Loaded {len(X_test)} test samples")
    
    # Pad sequences
    X_train_full = pad_sequences(
        X_train_full,
        maxlen=max_length,
        padding='post',
        truncating='post'
    )
    
    X_test = pad_sequences(
        X_test,
        maxlen=max_length,
        padding='post',
        truncating='post'
    )
    
    # Split training data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=test_size,
        random_state=42,
        stratify=y_train_full
    )
    
    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def load_csv_dataset(url=None, filepath=None):
    """
    Load dataset from CSV file
    If filepath is provided, use it. Otherwise download from URL
    """
    import pandas as pd
    import requests
    from io import StringIO
    
    if filepath and os.path.exists(filepath):
        print(f"Loading dataset from {filepath}...")
        df = pd.read_csv(filepath)
    elif url:
        print(f"Downloading dataset from {url}...")
        response = requests.get(url)
        df = pd.read_csv(StringIO(response.text))
    else:
        raise ValueError("Either filepath or url must be provided")
    
    # Assume CSV has 'text' and 'label' columns (adjust as needed)
    if 'text' not in df.columns or 'label' not in df.columns:
        # Try common column names
        text_col = df.columns[0] if 'review' in df.columns[0].lower() or 'text' in df.columns[0].lower() else df.columns[0]
        label_col = df.columns[1] if len(df.columns) > 1 else df.columns[-1]
        texts = df[text_col].astype(str).tolist()
        labels = df[label_col].astype(int).tolist()
    else:
        texts = df['text'].astype(str).tolist()
        labels = df['label'].astype(int).tolist()
    
    return texts, labels


def main():
    """Main training function with real dataset"""
    print("=" * 70)
    print("Sentiment Analysis RNN - Training with Real Dataset")
    print("=" * 70)
    
    start_time = time.time()
    
    # Load IMDB dataset (built-in, fast to load)
    print("\n1. Loading dataset...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_imdb_data(
        num_words=2000,
        max_length=200
    )
    
    # Initialize model with optimized hyperparameters
    print("\n2. Initializing RNN model with optimized hyperparameters...")
    model = FastSentimentRNN(
        vocab_size=2000,
        embedding_dim=128,
        max_length=200,
        lstm_units=128
    )
    
    # Build model
    print("\n3. Building model architecture...")
    model.build_model()
    # Build with sample input to get parameter count
    model.model.build(input_shape=(None, 200))
    print(f"   Total parameters: {model.model.count_params():,}")
    
    # Train model (optimized for speed)
    print("\n4. Training model (optimized for speed)...")
    print("   Using: batch_size=128, early stopping with patience=2")
    history = model.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=15,  # Reduced epochs for speed
        batch_size=128  # Larger batch for speed
    )
    
    # Evaluate on test set
    print("\n5. Evaluating on test set...")
    test_loss, test_accuracy, test_precision, test_recall = model.model.evaluate(
        X_test, y_test, verbose=0, batch_size=128
    )
    
    f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)
    
    print(f"\n   Test Results:")
    print(f"   Accuracy:  {test_accuracy:.4f}")
    print(f"   Precision: {test_precision:.4f}")
    print(f"   Recall:    {test_recall:.4f}")
    print(f"   F1 Score:  {f1_score:.4f}")
    
    # Make sample predictions
    print("\n6. Making sample predictions...")
    
    # Get word index for decoding
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
    
    # Decode some test samples
    def decode_review(encoded_review):
        return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review if i > 3])
    
    # Sample predictions
    sample_indices = [0, 1, 2, 3, 4]
    predictions = model.model.predict(X_test[sample_indices], verbose=0)
    
    print("\n   Sample Predictions:")
    for idx, pred in zip(sample_indices, predictions):
        sentiment = "Positive" if pred[0] >= 0.5 else "Negative"
        actual = "Positive" if y_test[idx] == 1 else "Negative"
        confidence = pred[0] if pred[0] >= 0.5 else 1 - pred[0]
        review_text = decode_review(X_test[idx])
        review_preview = review_text[:60] + "..." if len(review_text) > 60 else review_text
        print(f"\n   Review: '{review_preview}'")
        print(f"   Actual: {actual}, Predicted: {sentiment} (Confidence: {confidence:.4f})")
    
    # Save model
    print("\n7. Saving model...")
    # Create a dummy tokenizer for saving (IMDB uses pre-tokenized data)
    from tensorflow.keras.preprocessing.text import Tokenizer
    model.tokenizer = Tokenizer(num_words=2000)
    model.save_model()
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print("=" * 70)


if __name__ == "__main__":
    main()

