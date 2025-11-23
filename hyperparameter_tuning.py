"""
Hyperparameter Tuning for Sentiment Analysis RNN
Tests different combinations of vocab_size, embedding_dim, and lstm_units
to find the optimal configuration.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd
from itertools import product
import time
import os

class HyperparameterTuner:
    def __init__(self, texts, labels, max_length=100):
        """
        Initialize hyperparameter tuner
        
        Args:
            texts: List of training texts
            labels: List of training labels
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.results = []
        
    def create_model(self, vocab_size, embedding_dim, lstm_units):
        """Create a model with specified hyperparameters"""
        model = Sequential([
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                input_length=self.max_length
            ),
            Bidirectional(
                LSTM(
                    units=lstm_units,
                    return_sequences=False,
                    dropout=0.2,
                    recurrent_dropout=0.2
                )
            ),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train_and_evaluate(self, vocab_size, embedding_dim, lstm_units, 
                          X_train, y_train, X_val, y_val, epochs=10):
        """Train and evaluate a model with given hyperparameters"""
        print(f"\n{'='*60}")
        print(f"Testing: vocab_size={vocab_size}, embedding_dim={embedding_dim}, lstm_units={lstm_units}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Create model
        model = self.create_model(vocab_size, embedding_dim, lstm_units)
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=0
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate on validation set
        val_loss, val_accuracy, val_precision, val_recall = model.evaluate(
            X_val, y_val, verbose=0
        )
        
        training_time = time.time() - start_time
        
        # Calculate F1 score
        f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-7)
        
        # Count parameters
        total_params = model.count_params()
        
        result = {
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'lstm_units': lstm_units,
            'val_accuracy': val_accuracy,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'f1_score': f1_score,
            'val_loss': val_loss,
            'total_params': total_params,
            'training_time': training_time,
            'epochs_trained': len(history.history['loss'])
        }
        
        self.results.append(result)
        
        print(f"Results:")
        print(f"  Validation Accuracy: {val_accuracy:.4f}")
        print(f"  Validation Precision: {val_precision:.4f}")
        print(f"  Validation Recall: {val_recall:.4f}")
        print(f"  F1 Score: {f1_score:.4f}")
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Training Time: {training_time:.2f}s")
        print(f"  Epochs Trained: {len(history.history['loss'])}")
        
        # Clear session to free memory
        keras.backend.clear_session()
        
        return result
    
    def grid_search(self, vocab_sizes, embedding_dims, lstm_units_list, 
                   epochs=10, test_size=0.2):
        """
        Perform grid search over hyperparameter combinations
        
        Args:
            vocab_sizes: List of vocab_size values to test
            embedding_dims: List of embedding_dim values to test
            lstm_units_list: List of lstm_units values to test
            epochs: Maximum number of epochs per model
            test_size: Fraction of data for validation
        """
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING - GRID SEARCH")
        print("="*60)
        
        # Preprocess data once with maximum vocab_size
        max_vocab = max(vocab_sizes)
        tokenizer = Tokenizer(num_words=max_vocab, oov_token='<OOV>')
        tokenizer.fit_on_texts(self.texts)
        
        # Split data
        X_full, y_full = self._preprocess_with_tokenizer(tokenizer, max_vocab)
        X_train, X_val, y_train, y_val = train_test_split(
            X_full, y_full, test_size=test_size, random_state=42, stratify=y_full
        )
        
        print(f"\nTraining set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        
        # Grid search
        total_combinations = len(vocab_sizes) * len(embedding_dims) * len(lstm_units_list)
        print(f"\nTotal combinations to test: {total_combinations}")
        
        combination_num = 0
        for vocab_size, embedding_dim, lstm_units in product(
            vocab_sizes, embedding_dims, lstm_units_list
        ):
            combination_num += 1
            print(f"\n[{combination_num}/{total_combinations}]", end=" ")
            
            # Use the appropriate vocab_size for this combination
            X_train_trimmed, y_train_trimmed = self._preprocess_with_tokenizer(
                tokenizer, vocab_size
            )
            X_val_trimmed, y_val_trimmed = self._preprocess_with_tokenizer(
                tokenizer, vocab_size
            )
            
            # Split again to maintain same train/val split
            X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
                X_train_trimmed, y_train_trimmed, test_size=test_size, 
                random_state=42, stratify=y_train_trimmed
            )
            
            self.train_and_evaluate(
                vocab_size, embedding_dim, lstm_units,
                X_train_final, y_train_final, X_val_final, y_val_final,
                epochs=epochs
            )
        
        return self.results
    
    def _preprocess_with_tokenizer(self, tokenizer, vocab_size):
        """Preprocess data with a specific vocab_size"""
        # Create a new tokenizer with the desired vocab_size
        new_tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
        new_tokenizer.word_index = {k: v for k, v in tokenizer.word_index.items() 
                                   if v < vocab_size}
        new_tokenizer.word_index['<OOV>'] = 1
        
        sequences = new_tokenizer.texts_to_sequences(self.texts)
        padded = pad_sequences(sequences, maxlen=self.max_length, 
                             padding='post', truncating='post')
        labels = np.array(self.labels)
        return padded, labels
    
    def get_best_configuration(self, metric='f1_score'):
        """
        Get the best hyperparameter configuration
        
        Args:
            metric: Metric to optimize ('f1_score', 'val_accuracy', 'val_precision', 'val_recall')
        
        Returns:
            Best configuration dictionary
        """
        if not self.results:
            return None
        
        df = pd.DataFrame(self.results)
        best_idx = df[metric].idxmax()
        best_config = df.loc[best_idx].to_dict()
        
        return best_config
    
    def print_results_summary(self):
        """Print a summary of all results"""
        if not self.results:
            print("No results to display.")
            return
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING RESULTS SUMMARY")
        print("="*80)
        
        # Sort by F1 score
        df_sorted = df.sort_values('f1_score', ascending=False)
        
        print("\nTop 5 Configurations (by F1 Score):")
        print("-"*80)
        print(df_sorted[['vocab_size', 'embedding_dim', 'lstm_units', 
                         'val_accuracy', 'val_precision', 'val_recall', 
                         'f1_score', 'total_params', 'training_time']].head().to_string(index=False))
        
        print("\n" + "-"*80)
        print("Best Configuration:")
        print("-"*80)
        best = self.get_best_configuration('f1_score')
        for key, value in best.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Save results to CSV
        df_sorted.to_csv('hyperparameter_results.csv', index=False)
        print(f"\nAll results saved to 'hyperparameter_results.csv'")
        
        return df_sorted


def create_sample_data():
    """Create sample training data"""
    # Expanded sample data for better hyperparameter tuning
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
        "Love it! Great product and fast shipping.",
        "Superb quality and excellent customer support.",
        "Wonderful product, very pleased with the purchase.",
        "Top-notch quality, highly satisfied!",
        "Excellent value, great product overall.",
        "Outstanding service, product works great!",
        "Fantastic experience, would recommend to others.",
        "Great product, meets all expectations.",
        "Very satisfied, excellent quality product.",
        "Perfect product, exactly as described.",
        "Amazing purchase, very happy with it!"
    ]
    
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
        "Very poor quality, avoid this product.",
        "Awful product, complete waste of money.",
        "Disappointing quality, not as expected.",
        "Poor service and defective product.",
        "Terrible purchase, regret buying this.",
        "Low quality product, avoid at all costs.",
        "Very bad experience, product failed quickly.",
        "Poor quality control, defective item received.",
        "Not worth the price, poor quality.",
        "Disappointed with the product quality.",
        "Terrible value, product broke immediately."
    ]
    
    texts = positive_texts + negative_texts
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)
    
    return texts, labels


def main():
    """Main function for hyperparameter tuning"""
    print("="*60)
    print("Sentiment Analysis RNN - Hyperparameter Tuning")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    texts, labels = create_sample_data()
    print(f"   Loaded {len(texts)} samples")
    print(f"   Positive: {sum(labels)}, Negative: {len(labels) - sum(labels)}")
    
    # Define hyperparameter search space
    vocab_sizes = [1000, 2000, 3000]
    embedding_dims = [64, 128, 256]
    lstm_units_list = [32, 64, 128]
    
    print("\n2. Hyperparameter Search Space:")
    print(f"   vocab_size: {vocab_sizes}")
    print(f"   embedding_dim: {embedding_dims}")
    print(f"   lstm_units: {lstm_units_list}")
    print(f"   Total combinations: {len(vocab_sizes) * len(embedding_dims) * len(lstm_units_list)}")
    
    # Initialize tuner
    tuner = HyperparameterTuner(texts, labels, max_length=100)
    
    # Perform grid search
    print("\n3. Starting grid search...")
    results = tuner.grid_search(
        vocab_sizes=vocab_sizes,
        embedding_dims=embedding_dims,
        lstm_units_list=lstm_units_list,
        epochs=15,
        test_size=0.2
    )
    
    # Print summary
    print("\n4. Generating results summary...")
    df_results = tuner.print_results_summary()
    
    # Get best configuration
    best_config = tuner.get_best_configuration('f1_score')
    
    print("\n" + "="*60)
    print("RECOMMENDED HYPERPARAMETERS")
    print("="*60)
    print(f"vocab_size = {int(best_config['vocab_size'])}")
    print(f"embedding_dim = {int(best_config['embedding_dim'])}")
    print(f"lstm_units = {int(best_config['lstm_units'])}")
    print(f"\nExpected Performance:")
    print(f"  Accuracy: {best_config['val_accuracy']:.4f}")
    print(f"  F1 Score: {best_config['f1_score']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()

