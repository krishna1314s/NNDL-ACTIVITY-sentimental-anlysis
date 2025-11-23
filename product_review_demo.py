"""
Product Review Sentiment Analysis - Demo
Quick demonstration of product review classification
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

print("=" * 70)
print("Product Review Sentiment Analysis - Demo")
print("=" * 70)

# Use the better IMDB model if available
if os.path.exists('fast_sentiment_model.h5'):
    print("\n1. Using pre-trained IMDB model (better performance)...")
    # Use the fast_sentiment_model if available
    if os.path.exists('fast_sentiment_model.h5'):
        model = keras.models.load_model('fast_sentiment_model.h5')
        from tensorflow.keras.datasets import imdb
        word_index = imdb.get_word_index()
        max_length = 200
        vocab_size = 2000
        
        def encode_text(text, word_index, max_words=2000):
            words = text.lower().split()
            encoded = []
            for word in words:
                if word in word_index and word_index[word] < max_words:
                    encoded.append(word_index[word] + 3)
            return encoded if encoded else [1]
        
        tokenizer = None  # Will use encode_text function
        print("   Loaded IMDB-trained model (535K parameters)")
    else:
        print("   No model found. Please run train_fast.py first.")
        exit(1)

# Product review examples
print("\n2. Analyzing Product Reviews...")
print("=" * 70)

product_reviews = [
    "I absolutely love this product! It's amazing and works perfectly! Highly recommend to everyone!",
    "This is the worst purchase I've ever made. Terrible quality and broke immediately. Waste of money!",
    "The product is okay, nothing special but it works as expected. Average quality.",
    "Outstanding quality and excellent value for money! Very satisfied with my purchase!",
    "Poor quality, waste of money. Would not recommend to anyone. Very disappointed!",
    "Fantastic product! Great value and fast delivery! Exceeded my expectations!",
    "Very disappointed with this product. It doesn't work as described. Poor quality!",
    "Amazing experience! This exceeded all my expectations. Best purchase ever!",
    "The product is decent but could be better. Average quality, nothing special.",
    "Terrible experience. Product failed after one day of use. Very poor quality!"
]

predictions_list = []

for i, review in enumerate(product_reviews, 1):
    if tokenizer:
        # Use tokenizer
        seq = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    else:
        # Use word_index encoding
        encoded = encode_text(review, word_index, vocab_size)
        padded = pad_sequences([encoded], maxlen=max_length, padding='post', truncating='post')
    
    pred = model.predict(padded, verbose=0)[0][0]
    cls = "POSITIVE" if pred >= 0.5 else "NEGATIVE"
    confidence = pred if pred >= 0.5 else (1 - pred)
    confidence_pct = confidence * 100
    
    predictions_list.append((review, cls, confidence, pred))
    
    # Display result
    print(f"\n[{i}] {cls} ({confidence_pct:.1f}% confidence)")
    print(f"    Review: \"{review[:75]}{'...' if len(review) > 75 else ''}\"")
    print(f"    Raw Score: {pred:.4f}")

# Summary
print("\n" + "=" * 70)
print("Product Review Analysis Summary")
print("=" * 70)

positive_count = sum([1 for _, s, _, _ in predictions_list if s == "POSITIVE"])
negative_count = len(predictions_list) - positive_count
avg_confidence = np.mean([c for _, _, c, _ in predictions_list])

print(f"\nTotal Reviews Analyzed: {len(product_reviews)}")
print(f"Positive Reviews: {positive_count}")
print(f"Negative Reviews: {negative_count}")
print(f"Average Confidence: {avg_confidence*100:.1f}%")

# Show examples
print("\n" + "=" * 70)
print("Example Classifications:")
print("=" * 70)

print("\nPositive Review Example:")
pos_examples = [(r, c) for r, s, c, _ in predictions_list if s == "POSITIVE"]
if pos_examples:
    pos_example = max(pos_examples, key=lambda x: x[1])
    print(f"   Review: \"{pos_example[0][:70]}...\"")
    print(f"   Sentiment: POSITIVE ({pos_example[1]*100:.1f}% confidence)")

print("\nNegative Review Example:")
neg_examples = [(r, c) for r, s, c, _ in predictions_list if s == "NEGATIVE"]
if neg_examples:
    neg_example = max(neg_examples, key=lambda x: x[1])
    print(f"   Review: \"{neg_example[0][:70]}...\"")
    print(f"   Sentiment: NEGATIVE ({neg_example[1]*100:.1f}% confidence)")

print("\n" + "=" * 70)
print("Product Review Sentiment Analysis Complete!")
print("=" * 70)

