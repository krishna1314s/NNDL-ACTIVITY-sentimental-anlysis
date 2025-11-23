# Hyperparameter Tuning Results

## Summary

This document summarizes the hyperparameter tuning results for the RNN-based sentiment analysis model.

## Tested Configurations

Five different hyperparameter configurations were tested:

| Config | vocab_size | embedding_dim | lstm_units | Accuracy | F1 Score | Parameters |
|--------|------------|---------------|------------|----------|----------|------------|
| 1      | 1000       | 64            | 32         | 0.6250   | 0.6667   | 93,057     |
| 2      | 2000       | 128           | 64         | 0.7500   | 0.6667   | 363,137    |
| 3      | 3000       | 256           | 128        | 0.7500   | 0.8000   | 1,178,753  |
| 4      | 2000       | 256           | 64         | 0.6250   | 0.5714   | 684,673    |
| 5      | 2000       | 128           | 128        | 0.7500   | 0.8000   | 535,681    |

## Best Configuration

**Winner: Config 5** (Most Efficient)
- **vocab_size**: 2000
- **embedding_dim**: 128
- **lstm_units**: 128
- **Accuracy**: 0.7500
- **F1 Score**: 0.8000
- **Total Parameters**: 535,681

**Alternative: Config 3** (Higher Capacity)
- **vocab_size**: 3000
- **embedding_dim**: 256
- **lstm_units**: 128
- **Accuracy**: 0.7500
- **F1 Score**: 0.8000
- **Total Parameters**: 1,178,753

## Key Findings

1. **LSTM Units Impact**: Increasing LSTM units from 64 to 128 significantly improved F1 score (Config 2 vs Config 5).

2. **Vocabulary Size**: 
   - Config 3 (vocab_size=3000) achieved same performance as Config 5 (vocab_size=2000)
   - Config 5 is more efficient with fewer parameters

3. **Embedding Dimension**:
   - Config 4 (embedding_dim=256) performed worse than Config 5 (embedding_dim=128)
   - Higher embedding dimensions don't always improve performance

4. **Efficiency**: Config 5 provides the best balance between performance and model size.

## Recommendations

### For Production Use:
- **Use Config 5**: `vocab_size=2000, embedding_dim=128, lstm_units=128`
  - Best F1 score (0.8000)
  - Reasonable model size (535K parameters)
  - Good balance of performance and efficiency

### For Maximum Performance:
- **Use Config 3**: `vocab_size=3000, embedding_dim=256, lstm_units=128`
  - Same F1 score as Config 5
  - Larger model (1.18M parameters)
  - May require more computational resources

### For Resource-Constrained Environments:
- **Use Config 1**: `vocab_size=1000, embedding_dim=64, lstm_units=32`
  - Smallest model (93K parameters)
  - Lower performance (F1: 0.6667)
  - Fastest training and inference

## Updated Defaults

The main script (`sentiment_analysis_rnn.py`) has been updated with the optimized hyperparameters:
- `vocab_size = 2000` (previously 10000)
- `embedding_dim = 128` (unchanged)
- `max_length = 100` (previously 200)
- `lstm_units = 128` (previously 64)

## Next Steps

1. Test on larger datasets to validate these hyperparameters
2. Consider using pre-trained word embeddings (Word2Vec, GloVe) for better performance
3. Experiment with different architectures (GRU, stacked LSTM layers)
4. Fine-tune learning rate and batch size for further optimization

## Files

- `hyperparameter_tuning.py`: Full grid search implementation
- `compare_hyperparameters.py`: Quick comparison script
- `sentiment_analysis_rnn.py`: Main implementation (updated with best hyperparameters)

