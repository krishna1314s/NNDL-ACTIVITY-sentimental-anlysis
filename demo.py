"""
Quick demo script to test sentiment analysis predictions
"""

from sentiment_analysis_rnn import SentimentRNN
import numpy as np

def demo():
    """Quick demonstration of sentiment analysis"""
    
    print("=" * 60)
    print("Sentiment Analysis RNN - Quick Demo")
    print("=" * 60)
    
    # Sample training data (minimal for quick demo)
    print("\n1. Setting up minimal training data...")
    texts = [
        # Positive
        "I love this product!",
        "Great quality and fast delivery.",
        "Excellent service, highly recommend!",
        "Amazing product, worth every penny.",
        "Fantastic! Best purchase ever.",
        
        # Negative
        "Terrible product, broke immediately.",
        "Poor quality, waste of money.",
        "Very disappointed with this.",
        "Bad service and slow delivery.",
        "Not worth it, avoid this product."
    ]
    
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    
    # Initialize and train model
    print("\n2. Initializing RNN model...")
    model = SentimentRNN(
        vocab_size=1000,
        embedding_dim=64,
        max_length=50,
        lstm_units=32
    )
    
    print("\n3. Preprocessing data...")
    X, y = model.preprocess_data(texts, labels, fit_tokenizer=True)
    
    print("\n4. Building and training model (quick training)...")
    model.build_model()
    model.train(X, y, epochs=5, batch_size=4, validation_split=0.2)
    
    # Test predictions
    print("\n5. Testing predictions on new texts:")
    print("-" * 60)
    
    test_texts = [
        "I absolutely love this! It's amazing!",
        "This is the worst thing I've ever bought.",
        "The product is okay, nothing special.",
        "Outstanding quality and great value!",
        "Terrible experience, would not recommend."
    ]
    
    predictions = model.predict(test_texts)
    classes = model.predict_class(test_texts)
    
    for text, prob, cls in zip(test_texts, predictions, classes):
        sentiment = "POSITIVE" if cls == 1 else "NEGATIVE"
        confidence = prob if cls == 1 else (1 - prob)
        print(f"\nText: '{text}'")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Raw Score: {prob:.4f}")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)

if __name__ == "__main__":
    demo()

