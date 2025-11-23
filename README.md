# Sentiment Analysis using RNN (LSTM)

This project implements a Recurrent Neural Network (RNN) using Long Short-Term Memory (LSTM) cells for sentiment analysis. The model can classify text as having positive or negative sentiment.

## Features

- **Bidirectional LSTM**: Processes text sequences in both directions for better context understanding
- **Word Embeddings**: Converts words to dense vector representations
- **Dropout Regularization**: Prevents overfitting during training
- **Early Stopping**: Automatically stops training when validation loss stops improving
- **Model Checkpointing**: Saves the best model during training

## Architecture

The model consists of:
1. **Embedding Layer**: Converts word indices to dense vectors (128 dimensions)
2. **Bidirectional LSTM Layer**: 64 LSTM units processing sequences bidirectionally
3. **Dense Layer**: 64 neurons with ReLU activation
4. **Dropout Layer**: 50% dropout for regularization
5. **Output Layer**: Single neuron with sigmoid activation for binary classification

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the main script to train the model on sample data:

```bash
python sentiment_analysis_rnn.py
```

### Using the Model in Your Code

```python
from sentiment_analysis_rnn import SentimentRNN

# Initialize the model
rnn_model = SentimentRNN(
    vocab_size=5000,
    embedding_dim=128,
    max_length=100,
    lstm_units=64
)

# Load your training data
texts = ["I love this!", "This is terrible", ...]
labels = [1, 0, ...]  # 1 for positive, 0 for negative

# Preprocess data
X, y = rnn_model.preprocess_data(texts, labels, fit_tokenizer=True)

# Build and train the model
rnn_model.build_model()
rnn_model.train(X, y, epochs=10, batch_size=32)

# Make predictions
predictions = rnn_model.predict("I love this product!")
print(f"Sentiment score: {predictions[0]:.4f}")

# Predict class
sentiment = rnn_model.predict_class("This is great!")
print(f"Sentiment: {'Positive' if sentiment[0] == 1 else 'Negative'}")

# Save the model
rnn_model.save_model()
```

### Loading a Pre-trained Model

```python
from sentiment_analysis_rnn import SentimentRNN

# Initialize model
rnn_model = SentimentRNN()

# Load saved model and tokenizer
rnn_model.load_model('sentiment_rnn_model.h5', 'tokenizer.pkl')

# Use for predictions
predictions = rnn_model.predict("Your text here")
```

## Model Parameters

- **vocab_size**: Maximum number of words in vocabulary (default: 10000)
- **embedding_dim**: Dimension of word embeddings (default: 128)
- **max_length**: Maximum length of input sequences (default: 200)
- **lstm_units**: Number of LSTM units (default: 64)
- **num_classes**: Number of output classes (1 for binary classification)

## Training on Your Own Data

To train on your own dataset:

1. Prepare your data as lists of texts and labels:
   ```python
   texts = ["text1", "text2", ...]
   labels = [1, 0, 1, ...]  # 1 = positive, 0 = negative
   ```

2. Use the `SentimentRNN` class to preprocess, train, and evaluate:
   ```python
   rnn_model = SentimentRNN()
   X, y = rnn_model.preprocess_data(texts, labels)
   rnn_model.build_model()
   rnn_model.train(X, y, epochs=20)
   ```

## Dataset Recommendations

For better results, use larger datasets such as:
- **IMDB Movie Reviews**: 50,000 movie reviews
- **Amazon Product Reviews**: Large collection of product reviews
- **Twitter Sentiment Dataset**: Social media sentiment data

You can download these datasets from:
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)

## Model Performance

The model's performance depends on:
- Size and quality of training data
- Hyperparameter tuning
- Text preprocessing (cleaning, normalization)
- Vocabulary size and sequence length

## Files

- `sentiment_analysis_rnn.py`: Main implementation file
- `requirements.txt`: Python dependencies
- `README.md`: This file

## Output Files

After training, the following files will be created:
- `sentiment_rnn_model.h5`: Saved model weights
- `tokenizer.pkl`: Saved tokenizer for text preprocessing
- `best_sentiment_model.h5`: Best model checkpoint during training

## Notes

- The current implementation uses sample data for demonstration
- For production use, train on a larger, more diverse dataset
- Consider using pre-trained word embeddings (Word2Vec, GloVe) for better performance
- Experiment with hyperparameters to optimize for your specific use case

## License

This project is open source and available for educational purposes.

