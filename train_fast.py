"""
Fast Sentiment Analysis Training with Real Dataset
Uses IMDB dataset subset for quick training
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import imdb
import time

print("=" * 70)
print("Fast Sentiment Analysis Training")
print("=" * 70)

start_time = time.time()

# Load IMDB dataset (smaller subset for speed)
print("\n1. Loading IMDB dataset (subset for speed)...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=2000)

# Use subset for faster training
train_size = 5000  # Use 5000 samples instead of 25000
test_size = 1000   # Use 1000 samples instead of 25000

X_train = X_train[:train_size]
y_train = y_train[:train_size]
X_test = X_test[:test_size]
y_test = y_test[:test_size]

print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")

# Pad sequences
max_length = 200
X_train = pad_sequences(X_train, maxlen=max_length, padding='post', truncating='post')
X_test = pad_sequences(X_test, maxlen=max_length, padding='post', truncating='post')

# Split training into train/val
split_idx = int(len(X_train) * 0.8)
X_val = X_train[split_idx:]
y_val = y_train[split_idx:]
X_train = X_train[:split_idx]
y_train = y_train[:split_idx]

print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Build model with optimized hyperparameters
print("\n2. Building model...")
model = Sequential([
    Embedding(2000, 128, input_length=max_length),
    Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

# Build model to get parameter count
model.build(input_shape=(None, max_length))
print(f"   Total parameters: {model.count_params():,}")

# Train with fast settings
print("\n3. Training (fast mode)...")
print("   Batch size: 128, Epochs: 5, Early stopping: patience=2")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=128,
    callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=1)],
    verbose=1
)

# Evaluate
print("\n4. Evaluating on test set...")
test_loss, test_acc, test_prec, test_rec = model.evaluate(X_test, y_test, verbose=0, batch_size=128)
f1 = 2 * (test_prec * test_rec) / (test_prec + test_rec + 1e-7)

print(f"\n   Test Accuracy:  {test_acc:.4f}")
print(f"   Test Precision: {test_prec:.4f}")
print(f"   Test Recall:    {test_rec:.4f}")
print(f"   F1 Score:      {f1:.4f}")

# Sample predictions
print("\n5. Sample predictions:")
word_index = imdb.get_word_index()
reverse_word_index = {v: k for k, v in word_index.items()}

def decode(encoded):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded if i > 3])

samples = [0, 1, 2]
preds = model.predict(X_test[samples], verbose=0)

for i, idx in enumerate(samples):
    text = decode(X_test[idx])
    preview = text[:50] + "..." if len(text) > 50 else text
    actual = "Positive" if y_test[idx] == 1 else "Negative"
    pred = "Positive" if preds[i][0] >= 0.5 else "Negative"
    conf = preds[i][0] if preds[i][0] >= 0.5 else 1 - preds[i][0]
    print(f"\n   Review: '{preview}'")
    print(f"   Actual: {actual}, Predicted: {pred} ({conf:.2%})")

# Save model
print("\n6. Saving model...")
model.save('fast_sentiment_model.h5')
print("   Model saved to 'fast_sentiment_model.h5'")

elapsed = time.time() - start_time
print("\n" + "=" * 70)
print(f"Training completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
print("=" * 70)

