"""
Sentiment Analysis using RNN (LSTM)
This script implements a Recurrent Neural Network for sentiment analysis
using LSTM cells to classify text as positive or negative sentiment.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import pickle
import os

class SentimentRNN:
    def __init__(self, vocab_size=2000, embedding_dim=128, max_length=100, 
                 lstm_units=128, num_classes=1):
        """
        Initialize the Sentiment RNN model
        
        Args:
            vocab_size: Maximum number of words in vocabulary
            embedding_dim: Dimension of word embeddings
            max_length: Maximum length of input sequences
            lstm_units: Number of LSTM units
            num_classes: Number of output classes (1 for binary, 2 for multi-class)
        """
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
            # Embedding layer: converts word indices to dense vectors
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_length,
                name='embedding'
            ),
            
            # Bidirectional LSTM layer: processes sequences in both directions
            Bidirectional(
                LSTM(
                    units=self.lstm_units,
                    return_sequences=False,
                    dropout=0.2,
                    recurrent_dropout=0.2,
                    name='lstm'
                )
            ),
            
            # Dense layer with dropout for regularization
            Dense(64, activation='relu', name='dense_1'),
            Dropout(0.5, name='dropout'),
            
            # Output layer: sigmoid for binary classification
            Dense(self.num_classes, activation='sigmoid', name='output')
        ])
        
        # Compile the model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return self.model
    
    def preprocess_data(self, texts, labels=None, fit_tokenizer=True):
        """
        Preprocess text data for RNN input
        
        Args:
            texts: List of text strings
            labels: List of labels (optional, for training)
            fit_tokenizer: Whether to fit tokenizer on this data
        
        Returns:
            Processed sequences and labels (if provided)
        """
        if fit_tokenizer or self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
            self.tokenizer.fit_on_texts(texts)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Pad sequences to fixed length
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_length,
            padding='post',
            truncating='post'
        )
        
        if labels is not None:
            labels = np.array(labels)
            return padded_sequences, labels
        
        return padded_sequences
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=10, batch_size=32, validation_split=0.2):
        """
        Train the RNN model
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks for training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'best_sentiment_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Prepare validation data
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = None
        else:
            validation_data = None
        
        # Train the model
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, texts):
        """
        Predict sentiment for given texts
        
        Args:
            texts: List of text strings or single text string
        
        Returns:
            Predicted probabilities (0-1, where 1 is positive sentiment)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess texts
        sequences = self.preprocess_data(texts, fit_tokenizer=False)
        
        # Make predictions
        predictions = self.model.predict(sequences, verbose=0)
        
        return predictions.flatten()
    
    def predict_class(self, texts, threshold=0.5):
        """
        Predict sentiment class (positive/negative)
        
        Args:
            texts: List of text strings or single text string
            threshold: Classification threshold
        
        Returns:
            List of predicted classes (0=negative, 1=positive)
        """
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
    
    def load_model(self, model_path='sentiment_rnn_model.h5', tokenizer_path='tokenizer.pkl'):
        """Load a trained model and tokenizer"""
        if os.path.exists(model_path):
            self.model = keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            print(f"Tokenizer loaded from {tokenizer_path}")


def create_sample_data():
    """
    Create sample training data for demonstration
    In practice, you would load your own dataset
    """
    # Sample positive reviews
    positive_texts = [
        "I love this product! It's amazing and works perfectly.",
        "Great quality, highly recommend to everyone.",
        "Excellent service and fast delivery. Very satisfied!",
        "This is the best purchase I've made. Worth every penny.",
        "Outstanding quality and great value for money.",
        "Fantastic product, exceeded my expectations.",
        "Very happy with this purchase. Great customer service.",
        "Perfect! Exactly what I was looking for.",
        "Amazing quality, would definitely buy again.",
        "Love it! Great product and fast shipping."
    ]
    
    # Sample negative reviews
    negative_texts = [
        "Terrible product, broke after one day.",
        "Poor quality, not worth the money.",
        "Very disappointed with this purchase.",
        "Waste of money, doesn't work as described.",
        "Bad quality and slow delivery.",
        "Not satisfied at all. Poor customer service.",
        "This product is a complete disappointment.",
        "Low quality, would not recommend.",
        "Terrible experience, product doesn't work.",
        "Very poor quality, avoid this product."
    ]
    
    # Combine and create labels
    texts = positive_texts + negative_texts
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)
    
    return texts, labels


def main():
    """Main function to demonstrate sentiment analysis with RNN"""
    print("=" * 60)
    print("Sentiment Analysis using RNN (LSTM)")
    print("=" * 60)
    
    # Create or load your dataset
    print("\n1. Loading data...")
    texts, labels = create_sample_data()
    print(f"   Loaded {len(texts)} samples")
    print(f"   Positive samples: {sum(labels)}")
    print(f"   Negative samples: {len(labels) - sum(labels)}")
    
    # Initialize the model with optimized hyperparameters
    print("\n2. Initializing RNN model...")
    rnn_model = SentimentRNN(
        vocab_size=2000,
        embedding_dim=128,
        max_length=100,
        lstm_units=128
    )
    
    # Preprocess data
    print("\n3. Preprocessing data...")
    X, y = rnn_model.preprocess_data(texts, labels, fit_tokenizer=True)
    print(f"   Sequence shape: {X.shape}")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Build model
    print("\n4. Building model architecture...")
    model = rnn_model.build_model()
    model.summary()
    
    # Train the model
    print("\n5. Training the model...")
    history = rnn_model.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=20,
        batch_size=16
    )
    
    # Evaluate the model
    print("\n6. Evaluating the model...")
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
        X_test, y_test, verbose=0
    )
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    print(f"   Test Precision: {test_precision:.4f}")
    print(f"   Test Recall: {test_recall:.4f}")
    
    # Make predictions on sample texts
    print("\n7. Making predictions on sample texts...")
    sample_texts = [
        "I absolutely love this product! It's fantastic!",
        "This is terrible, worst purchase ever.",
        "The product is okay, nothing special.",
        "Amazing quality and great value!"
    ]
    
    predictions = rnn_model.predict(sample_texts)
    classes = rnn_model.predict_class(sample_texts)
    
    print("\n   Predictions:")
    for text, prob, cls in zip(sample_texts, predictions, classes):
        sentiment = "Positive" if cls == 1 else "Negative"
        print(f"   Text: '{text[:50]}...'")
        print(f"   Sentiment: {sentiment} (Confidence: {prob:.4f})")
        print()
    
    # Save the model
    print("\n8. Saving model...")
    rnn_model.save_model()
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

