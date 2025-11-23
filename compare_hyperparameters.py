"""
Quick comparison of different hyperparameter configurations
This script tests a few key configurations to find the best one quickly.
"""

from sentiment_analysis_rnn import SentimentRNN
from sklearn.model_selection import train_test_split
import numpy as np

def create_sample_data():
    """Create sample training data"""
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


def test_configuration(vocab_size, embedding_dim, lstm_units, texts, labels, max_length=100):
    """Test a specific hyperparameter configuration"""
    print(f"\n{'='*70}")
    print(f"Testing: vocab_size={vocab_size}, embedding_dim={embedding_dim}, lstm_units={lstm_units}")
    print(f"{'='*70}")
    
    # Initialize model
    model = SentimentRNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        max_length=max_length,
        lstm_units=lstm_units
    )
    
    # Preprocess data
    X, y = model.preprocess_data(texts, labels, fit_tokenizer=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Build and train
    model.build_model()
    history = model.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=15,
        batch_size=16
    )
    
    # Evaluate
    test_loss, test_accuracy, test_precision, test_recall = model.model.evaluate(
        X_test, y_test, verbose=0
    )
    
    f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)
    total_params = model.model.count_params()
    
    result = {
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'lstm_units': lstm_units,
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': f1_score,
        'total_params': total_params
    }
    
    print(f"\nResults:")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1 Score: {f1_score:.4f}")
    print(f"  Total Parameters: {total_params:,}")
    
    return result


def main():
    """Compare different hyperparameter configurations"""
    print("="*70)
    print("QUICK HYPERPARAMETER COMPARISON")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    texts, labels = create_sample_data()
    print(f"Loaded {len(texts)} samples")
    
    # Define configurations to test
    configurations = [
        # Small model (fast, less capacity)
        {'vocab_size': 1000, 'embedding_dim': 64, 'lstm_units': 32},
        
        # Medium model (balanced)
        {'vocab_size': 2000, 'embedding_dim': 128, 'lstm_units': 64},
        
        # Large model (more capacity, slower)
        {'vocab_size': 3000, 'embedding_dim': 256, 'lstm_units': 128},
        
        # High embedding, moderate LSTM
        {'vocab_size': 2000, 'embedding_dim': 256, 'lstm_units': 64},
        
        # Moderate embedding, high LSTM
        {'vocab_size': 2000, 'embedding_dim': 128, 'lstm_units': 128},
    ]
    
    results = []
    
    for i, config in enumerate(configurations, 1):
        print(f"\n[{i}/{len(configurations)}] ", end="")
        result = test_configuration(
            config['vocab_size'],
            config['embedding_dim'],
            config['lstm_units'],
            texts, labels
        )
        results.append(result)
    
    # Print summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Config':<20} {'Accuracy':<12} {'F1 Score':<12} {'Params':<15}")
    print("-"*70)
    
    for i, result in enumerate(results, 1):
        config_str = f"Config {i}"
        print(f"{config_str:<20} {result['accuracy']:<12.4f} {result['f1_score']:<12.4f} {result['total_params']:<15,}")
    
    # Find best
    best_idx = max(range(len(results)), key=lambda i: results[i]['f1_score'])
    best = results[best_idx]
    
    print("\n" + "="*70)
    print("BEST CONFIGURATION")
    print("="*70)
    print(f"vocab_size = {best['vocab_size']}")
    print(f"embedding_dim = {best['embedding_dim']}")
    print(f"lstm_units = {best['lstm_units']}")
    print(f"\nPerformance:")
    print(f"  Accuracy: {best['accuracy']:.4f}")
    print(f"  F1 Score: {best['f1_score']:.4f}")
    print(f"  Total Parameters: {best['total_params']:,}")
    print("="*70)


if __name__ == "__main__":
    main()

