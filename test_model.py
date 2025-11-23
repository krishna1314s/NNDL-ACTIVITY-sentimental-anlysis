"""
Test the trained sentiment analysis model with sample texts
"""

from sentiment_analysis_rnn import SentimentRNN
import numpy as np

print("=" * 70)
print("Sentiment Analysis Model - Testing Predictions")
print("=" * 70)

# Initialize model
print("\n1. Loading trained model...")
model = SentimentRNN(
    vocab_size=2000,
    embedding_dim=128,
    max_length=100,
    lstm_units=128
)

# Try to load saved model
try:
    model.load_model('sentiment_rnn_model.h5', 'tokenizer.pkl')
    print("   Model loaded successfully!")
except:
    print("   No saved model found. Training a quick model...")
    # Quick training with sample data
    from sentiment_analysis_rnn import create_sample_data
    texts, labels = create_sample_data()
    X, y = model.preprocess_data(texts, labels, fit_tokenizer=True)
    model.build_model()
    model.train(X, y, epochs=5, batch_size=16)

# Test texts with different sentiments
test_texts = [
    "I absolutely love this product! It's amazing and works perfectly. Highly recommend!",
    "This is the worst purchase I've ever made. Terrible quality and broke immediately.",
    "The product is okay, nothing special but it works as expected.",
    "Outstanding quality and excellent value for money. Very satisfied with my purchase!",
    "Poor quality, waste of money. Would not recommend to anyone.",
    "Fantastic service and fast delivery. Great customer support!",
    "Very disappointed with this product. It doesn't work as described.",
    "Amazing experience! This exceeded all my expectations. Best buy ever!",
    "The product is decent but could be better. Average quality.",
    "Terrible experience. Product failed after one day of use. Very poor quality."
]

print("\n2. Making predictions on test texts...")
print("=" * 70)

predictions = model.predict(test_texts)
classes = model.predict_class(test_texts)

for i, (text, prob, cls) in enumerate(zip(test_texts, predictions, classes), 1):
    sentiment = "POSITIVE" if cls == 1 else "NEGATIVE"
    confidence = prob if cls == 1 else (1 - prob)
    confidence_pct = confidence * 100
    
    # Label
    if cls == 1:
        sentiment_label = "POSITIVE"
    else:
        sentiment_label = "NEGATIVE"
    
    print(f"\n[{i}] {sentiment_label} ({confidence_pct:.1f}% confidence)")
    print(f"    Text: \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
    print(f"    Raw Score: {prob:.4f}")

print("\n" + "=" * 70)
print("Summary Statistics")
print("=" * 70)

positive_count = sum(classes)
negative_count = len(classes) - positive_count
avg_confidence = np.mean([p if c == 1 else (1-p) for p, c in zip(predictions, classes)])

print(f"Total predictions: {len(test_texts)}")
print(f"Positive: {positive_count}")
print(f"Negative: {negative_count}")
print(f"Average confidence: {avg_confidence*100:.1f}%")
print("=" * 70)

