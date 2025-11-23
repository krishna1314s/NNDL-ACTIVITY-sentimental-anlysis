"""
Product Review Sentiment Analysis - Fast Training
Classifies customer product reviews as Positive or Negative
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import time

print("=" * 70)
print("Product Review Sentiment Analysis - Fast Training")
print("=" * 70)

start_time = time.time()

# Product review dataset
print("\n1. Loading Product Review Dataset...")
product_reviews = [
    # Positive Reviews
    "I love this product! It's amazing and works perfectly. Highly recommend!",
    "Great quality, highly recommend to everyone. Worth every penny!",
    "Excellent product and fast delivery. Very satisfied with my purchase!",
    "This is the best purchase I've made. Outstanding quality!",
    "Fantastic product, exceeded my expectations. Great value for money!",
    "Very happy with this purchase. Great customer service and quality!",
    "Perfect! Exactly what I was looking for. Amazing quality!",
    "Love it! Great product and fast shipping. Highly satisfied!",
    "Superb quality and excellent customer support. Very pleased!",
    "Wonderful product, very pleased with the purchase. Top quality!",
    "Outstanding service, product works great! Highly recommend!",
    "Fantastic experience, would recommend to others. Great product!",
    "Great product, meets all expectations. Excellent quality!",
    "Very satisfied, excellent quality product. Worth buying!",
    "Perfect product, exactly as described. Amazing purchase!",
    "Amazing purchase, very happy with it! Great quality!",
    "Excellent value, great product overall. Highly satisfied!",
    "Top-notch quality, highly satisfied! Great purchase!",
    "Outstanding product, works perfectly. Very happy!",
    "Fantastic quality, would definitely buy again. Great value!",
    
    # Negative Reviews
    "Terrible product, broke after one day. Poor quality!",
    "Poor quality, not worth the money. Very disappointed!",
    "Very disappointed with this purchase. Waste of money!",
    "Waste of money, doesn't work as described. Bad quality!",
    "Bad quality and slow delivery. Not satisfied at all!",
    "Not satisfied at all. Poor customer service and quality!",
    "This product is a complete disappointment. Low quality!",
    "Low quality, would not recommend. Terrible experience!",
    "Terrible experience, product doesn't work. Very poor quality!",
    "Very poor quality, avoid this product. Waste of money!",
    "Awful product, complete waste of money. Poor quality!",
    "Disappointing quality, not as expected. Very bad!",
    "Poor service and defective product. Not worth it!",
    "Terrible purchase, regret buying this. Low quality!",
    "Low quality product, avoid at all costs. Very bad!",
    "Very bad experience, product failed quickly. Poor quality!",
    "Poor quality control, defective item received. Disappointed!",
    "Not worth the price, poor quality. Very disappointed!",
    "Disappointed with the product quality. Terrible value!",
    "Terrible value, product broke immediately. Poor quality!"
]

labels = [1] * 20 + [0] * 20  # 20 positive, 20 negative

print(f"   Loaded {len(product_reviews)} product reviews")
print(f"   Positive reviews: {sum(labels)}")
print(f"   Negative reviews: {len(labels) - sum(labels)}")

# Preprocess data
print("\n2. Preprocessing data...")
vocab_size = 1000
max_length = 50

tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(product_reviews)

sequences = tokenizer.texts_to_sequences(product_reviews)
X = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
y = np.array(labels)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")

# Build model (optimized for speed)
print("\n3. Building model (optimized for speed)...")
model = Sequential([
    Embedding(vocab_size, 64, input_length=max_length),
    Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2)),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

model.build(input_shape=(None, max_length))
print(f"   Total parameters: {model.count_params():,}")

# Train model (fast settings)
print("\n4. Training model (fast mode)...")
print("   Batch size: 8, Epochs: 10, Early stopping: patience=2")

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=8,
    callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=1)],
    verbose=1
)

# Evaluate
print("\n5. Evaluating on test set...")
test_loss, test_acc, test_prec, test_rec = model.evaluate(X_test, y_test, verbose=0)
f1 = 2 * (test_prec * test_rec) / (test_prec + test_rec + 1e-7)

print(f"\n   Test Accuracy:  {test_acc:.4f}")
print(f"   Test Precision: {test_prec:.4f}")
print(f"   Test Recall:    {test_rec:.4f}")
print(f"   F1 Score:       {f1:.4f}")

# Test on new product reviews
print("\n6. Testing on New Product Reviews...")
print("=" * 70)

new_reviews = [
    "I absolutely love this product! It's amazing and works perfectly!",
    "This is the worst purchase I've ever made. Terrible quality!",
    "The product is okay, nothing special but it works.",
    "Outstanding quality and excellent value for money!",
    "Poor quality, waste of money. Would not recommend!",
    "Fantastic product! Great value and fast delivery!",
    "Very disappointed with this product. Doesn't work as described.",
    "Amazing experience! This exceeded all my expectations!",
    "The product is decent but could be better. Average quality.",
    "Terrible experience. Product failed after one day of use."
]

predictions = []
for review in new_reviews:
    seq = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    pred = model.predict(padded, verbose=0)[0][0]
    cls = "POSITIVE" if pred >= 0.5 else "NEGATIVE"
    confidence = pred if pred >= 0.5 else (1 - pred)
    predictions.append((review, cls, confidence, pred))

for i, (review, sentiment, conf, raw) in enumerate(predictions, 1):
    print(f"\n[{i}] {sentiment} ({conf*100:.1f}% confidence)")
    print(f"    Review: \"{review[:70]}{'...' if len(review) > 70 else ''}\"")
    print(f"    Raw Score: {raw:.4f}")

# Summary
print("\n" + "=" * 70)
print("Summary Statistics")
print("=" * 70)
positive_count = sum([1 for _, s, _, _ in predictions if s == "POSITIVE"])
negative_count = len(predictions) - positive_count
avg_confidence = np.mean([c for _, _, c, _ in predictions])

print(f"Total predictions: {len(predictions)}")
print(f"Positive: {positive_count}")
print(f"Negative: {negative_count}")
print(f"Average confidence: {avg_confidence*100:.1f}%")

# Save model
print("\n7. Saving model...")
model.save('product_review_model.h5')
print("   Model saved to 'product_review_model.h5'")

elapsed = time.time() - start_time
print("\n" + "=" * 70)
print(f"Training completed in {elapsed:.1f} seconds")
print("=" * 70)

