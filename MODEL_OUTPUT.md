# Sentiment Analysis RNN - Model Output and Results

## Model Performance

### Training Configuration
- **Dataset**: IMDB Movie Reviews (5,000 training samples, 1,000 test samples)
- **Model Architecture**: Bidirectional LSTM
- **Hyperparameters**:
  - vocab_size: 2000
  - embedding_dim: 128
  - max_length: 200
  - lstm_units: 128
  - Total Parameters: 535,681

### Performance Metrics
- **Test Accuracy**: 74.60%
- **Test Precision**: 68.77%
- **Test Recall**: 85.02%
- **F1 Score**: 76.04%

### Training Time
- Total training time: 5.4 minutes (322 seconds)
- Training samples: 4,000
- Validation samples: 1,000
- Test samples: 1,000

---

## Sample Predictions

### Test Text Predictions (10 samples)

#### 1. Positive Sentiment (91.6% confidence)
**Text**: "I absolutely love this movie! It's amazing and works perfectly. Highly recommend!"
- **Prediction**: POSITIVE
- **Confidence**: 91.6%
- **Raw Score**: 0.9161

#### 2. Negative Sentiment (98.1% confidence)
**Text**: "This is the worst film I've ever seen. Terrible quality and boring immediately."
- **Prediction**: NEGATIVE
- **Confidence**: 98.1%
- **Raw Score**: 0.0193

#### 3. Neutral/Positive Sentiment (51.5% confidence)
**Text**: "The movie is okay, nothing special but it works as expected."
- **Prediction**: POSITIVE
- **Confidence**: 51.5%
- **Raw Score**: 0.5150

#### 4. Positive Sentiment (88.6% confidence)
**Text**: "Outstanding quality and excellent value for money. Very satisfied with my purchase!"
- **Prediction**: POSITIVE
- **Confidence**: 88.6%
- **Raw Score**: 0.8863

#### 5. Negative Sentiment (92.4% confidence)
**Text**: "Poor quality, waste of money. Would not recommend to anyone."
- **Prediction**: NEGATIVE
- **Confidence**: 92.4%
- **Raw Score**: 0.0755

#### 6. Positive Sentiment (88.9% confidence)
**Text**: "Fantastic service and fast delivery. Great customer support!"
- **Prediction**: POSITIVE
- **Confidence**: 88.9%
- **Raw Score**: 0.8888

#### 7. Negative Sentiment (83.7% confidence)
**Text**: "Very disappointed with this product. It doesn't work as described."
- **Prediction**: NEGATIVE
- **Confidence**: 83.7%
- **Raw Score**: 0.1631

#### 8. Positive Sentiment (89.3% confidence)
**Text**: "Amazing experience! This exceeded all my expectations. Best buy ever!"
- **Prediction**: POSITIVE
- **Confidence**: 89.3%
- **Raw Score**: 0.8933

#### 9. Negative Sentiment (75.6% confidence)
**Text**: "The product is decent but could be better. Average quality."
- **Prediction**: NEGATIVE
- **Confidence**: 75.6%
- **Raw Score**: 0.2436

#### 10. Negative Sentiment (95.3% confidence)
**Text**: "Terrible experience. Product failed after one day of use. Very poor quality."
- **Prediction**: NEGATIVE
- **Confidence**: 95.3%
- **Raw Score**: 0.0472

---

## Summary Statistics

### Prediction Distribution
- **Total Predictions**: 10
- **Positive Predictions**: 5 (50%)
- **Negative Predictions**: 5 (50%)
- **Average Confidence**: 85.5%

### Confidence Analysis
- **High Confidence (>80%)**: 8 predictions
- **Medium Confidence (60-80%)**: 1 prediction
- **Low Confidence (<60%)**: 1 prediction

---

## Model Architecture Details

```
Model: Sequential
├── Embedding Layer
│   ├── Input: vocab_size=2000
│   ├── Output: embedding_dim=128
│   └── Input Length: 200
├── Bidirectional LSTM
│   ├── Units: 128
│   ├── Dropout: 0.2
│   └── Recurrent Dropout: 0.2
├── Dense Layer
│   ├── Units: 64
│   └── Activation: ReLU
├── Dropout Layer (0.5)
└── Output Layer
    ├── Units: 1
    └── Activation: Sigmoid
```

---

## Training History

### Epoch Performance
- **Epoch 1**: Training Accuracy: 51.28%, Validation Accuracy: 51.00%
- **Epoch 2**: Training Accuracy: 64.00%, Validation Accuracy: 69.20%
- **Epoch 3**: Training Accuracy: 74.83%, Validation Accuracy: 72.60%
- **Epoch 4**: Training Accuracy: 80.87%, Validation Accuracy: 75.30% (Best)
- **Epoch 5**: Training Accuracy: 82.95%, Validation Accuracy: 75.60%

**Best Model**: Saved at Epoch 4 with validation accuracy of 75.30%

---

## Hyperparameter Tuning Results

### Best Configuration Found
- **vocab_size**: 2000
- **embedding_dim**: 128
- **lstm_units**: 128
- **F1 Score**: 0.8000
- **Accuracy**: 0.7500

### Comparison of Configurations Tested

| Config | vocab_size | embedding_dim | lstm_units | Accuracy | F1 Score | Parameters |
|--------|------------|---------------|------------|----------|----------|------------|
| 1      | 1000       | 64            | 32         | 0.6250   | 0.6667   | 93,057     |
| 2      | 2000       | 128           | 64         | 0.7500   | 0.6667   | 363,137    |
| 3      | 3000       | 256           | 128        | 0.7500   | 0.8000   | 1,178,753  |
| 4      | 2000       | 256           | 64         | 0.6250   | 0.5714   | 684,673    |
| **5**  | **2000**   | **128**       | **128**    | **0.7500** | **0.8000** | **535,681** |

**Selected Configuration**: Config 5 (best balance of performance and efficiency)

---

## Usage Example

```python
from sentiment_analysis_rnn import SentimentRNN

# Initialize model
model = SentimentRNN(
    vocab_size=2000,
    embedding_dim=128,
    max_length=100,
    lstm_units=128
)

# Load trained model
model.load_model('fast_sentiment_model.h5', 'tokenizer.pkl')

# Make prediction
text = "I love this product!"
prediction = model.predict(text)
sentiment = "Positive" if prediction[0] >= 0.5 else "Negative"
print(f"Sentiment: {sentiment} ({prediction[0]:.2%} confidence)")
```

---

## Files Generated

- `fast_sentiment_model.h5` - Trained model weights
- `sentiment_rnn_model.h5` - Alternative model
- `tokenizer.pkl` - Text tokenizer
- `best_sentiment_model.h5` - Best model checkpoint
- `hyperparameter_results.csv` - Hyperparameter tuning results

---

## Conclusion

The RNN-based sentiment analysis model successfully classifies text sentiment with:
- **74.6% accuracy** on test data
- **76.04% F1 score**
- **High confidence** predictions (>85% average) for clear positive/negative texts
- **Efficient model size** (535K parameters)

The model performs well on clear positive and negative sentiments, with lower confidence on neutral or ambiguous texts, which is expected behavior.

