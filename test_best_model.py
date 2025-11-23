"""
Test the best trained model (trained on IMDB dataset)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

print("=" * 70)
print("Sentiment Analysis Model - Testing with Best Model")
print("=" * 70)

# Load the best model (trained on IMDB)
print("\n1. Loading best trained model (IMDB dataset)...")
try:
    model = keras.models.load_model('fast_sentiment_model.h5')
    print("   Model loaded successfully!")
    print(f"   Model parameters: {model.count_params():,}")
except Exception as e:
    print(f"   Error loading model: {e}")
    print("   Please run train_fast.py first to train the model")
    exit(1)

# Load word index for text encoding
print("\n2. Loading word index...")
word_index = imdb.get_word_index()
reverse_word_index = {v: k for k, v in word_index.items()}

def encode_text(text, word_index, max_words=2000):
    """Encode text to sequence"""
    words = text.lower().split()
    encoded = []
    for word in words:
        if word in word_index and word_index[word] < max_words:
            encoded.append(word_index[word] + 3)  # +3 because of special tokens
    return encoded

# Test texts
test_texts = [
    "I absolutely love this movie! It's amazing and works perfectly. Highly recommend!",
    "This is the worst film I've ever seen. Terrible quality and boring immediately.",
    "The movie is okay, nothing special but it works as expected.",
    "Outstanding quality and excellent value for money. Very satisfied with my purchase!",
    "Poor quality, waste of money. Would not recommend to anyone.",
    "Fantastic service and fast delivery. Great customer support!",
    "Very disappointed with this product. It doesn't work as described.",
    "Amazing experience! This exceeded all my expectations. Best buy ever!",
    "The product is decent but could be better. Average quality.",
    "Terrible experience. Product failed after one day of use. Very poor quality."
]

print("\n3. Making predictions on test texts...")
print("=" * 70)

max_length = 200
predictions_list = []

for i, text in enumerate(test_texts, 1):
    # Encode text
    encoded = encode_text(text, word_index)
    if len(encoded) == 0:
        encoded = [1]  # Use OOV token if no words found
    
    # Pad sequence
    padded = pad_sequences([encoded], maxlen=max_length, padding='post', truncating='post')
    
    # Predict
    pred = model.predict(padded, verbose=0)[0][0]
    cls = 1 if pred >= 0.5 else 0
    sentiment = "POSITIVE" if cls == 1 else "NEGATIVE"
    confidence = pred if cls == 1 else (1 - pred)
    confidence_pct = confidence * 100
    
    predictions_list.append((pred, cls))
    
    print(f"\n[{i}] {sentiment} ({confidence_pct:.1f}% confidence)")
    print(f"    Text: \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
    print(f"    Raw Score: {pred:.4f}")

print("\n" + "=" * 70)
print("Summary Statistics")
print("=" * 70)

positive_count = sum([c for _, c in predictions_list])
negative_count = len(predictions_list) - positive_count
avg_confidence = np.mean([p if c == 1 else (1-p) for p, c in predictions_list])

print(f"Total predictions: {len(test_texts)}")
print(f"Positive: {positive_count}")
print(f"Negative: {negative_count}")
print(f"Average confidence: {avg_confidence*100:.1f}%")
print("=" * 70)

