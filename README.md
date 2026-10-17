# Autoregressive Text Generation Using RNN, BiRNN, LSTM and GRU

## 1. Project Overview

This project implements and compares four recurrent neural network architectures for word-level autoregressive text generation:

- Simple Recurrent Neural Network (RNN)
- Bidirectional Recurrent Neural Network (BiRNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)

The objective is to predict the next word from a sequence of preceding words, generate text sequentially, evaluate model performance, and analyse how sampling temperature affects generated text.

## 2. Dataset

**Dataset:** *The Time Machine* by H. G. Wells

**Dataset source:** [Project Gutenberg — The Time Machine](https://www.gutenberg.org/ebooks/35)

**Plain-text download:** https://www.gutenberg.org/cache/epub/35/pg35.txt

The dataset is a public-domain science-fiction novel containing narrative descriptions, dialogue, and varied vocabulary. The text is normalized, tokenized, and converted into integer IDs. Rare words are mapped to an unknown token, and fixed-length input sequences are created with the subsequent word as the prediction target.

The data is divided chronologically into training and validation sets.

## 3. Methodology

### Data preprocessing

- Convert text to lowercase.
- Tokenize words, numbers, and punctuation.
- Build a vocabulary with a maximum size of 12,000 tokens.
- Map rare tokens to `<unk>`.
- Create input sequences of 20 tokens with the following token as the target.
- Use a 90% training split and a 10% validation split.

### Model architectures

**Simple RNN:** Processes the input sequence using a recurrent hidden state.

**Bidirectional RNN:** Processes the known input sequence in both directions before predicting the next word.

**LSTM:** Uses a cell state and gates to help learn longer-term dependencies.

**GRU:** Uses update and reset gates to control information flow with a simpler architecture than LSTM.

All four models use the same dataset split, embedding dimension, hidden-unit count, optimizer, batch size, and maximum epoch count for comparison.

## 4. Evaluation Metrics

The models are evaluated using:

- **Validation loss:** Measures next-word prediction error.
- **Perplexity:** Exponential of validation cross-entropy loss; lower values generally indicate better predictive performance.
- **Validation accuracy:** Fraction of next-word predictions that match the target.
- **Training time:** Time taken to train each model.
- **Lexical diversity:** Proportion of unique words in generated text.
- **Repeated bigram ratio:** Indicator of repeated adjacent word pairs.
- **Qualitative analysis:** Coherence, contextual consistency, grammatical quality, diversity, and repetition.

## 5. Experimental Results

The following results were obtained from the training run.

| Model | Validation Loss | Perplexity | Validation Accuracy | Training Time (s) |
|---|---:|---:|---:|---:|
| Simple RNN | 5.8962 | 363.67 | 12.62% | 50.17 |
| Bidirectional RNN | 5.9916 | 400.05 | 12.54% | 69.55 |
| GRU | 6.0156 | 409.76 | 13.64% | 77.23 |
| LSTM | 6.0562 | 426.76 | 10.11% | 38.80 |

### Model comparison

The Simple RNN achieved the lowest validation loss and perplexity, indicating the strongest predictive performance according to these metrics. The GRU achieved the highest validation accuracy at 13.64%. The LSTM trained fastest, taking approximately 38.80 seconds.

The results show that the model with the lowest loss is not necessarily the model with the highest accuracy or shortest training time. The overall choice depends on the performance criterion.

## 6. Sampling Temperature Analysis

Text was generated from the same seed sequences using temperatures of 0.5, 1.0, and 1.5.

- **Temperature 0.5:** Produces more conservative word choices. The generated text may become repetitive and overuse common phrases.
- **Temperature 1.0:** Provides a middle ground between predictable and varied word selection, although grammatical and contextual errors remain.
- **Temperature 1.5:** Encourages more varied word choices but can produce less coherent sentences and disconnected ideas.

The generated samples illustrate the trade-off between predictability and diversity. Higher lexical diversity does not necessarily imply better text quality.

## 7. Conclusion

The experiments demonstrate the differences between Simple RNN, Bidirectional RNN, LSTM, and GRU models for word-level autoregressive text generation.

The Simple RNN performed best in validation loss and perplexity, while GRU achieved the highest validation accuracy. LSTM had the shortest training time in this experiment. Based on these measurements, Simple RNN is the strongest candidate for predictive performance, but generated-text quality must also be considered when choosing an overall model.

Temperature sampling demonstrated that conservative sampling can increase repetition, while higher temperatures can improve diversity at the expense of coherence. The experiment also highlights the difficulty of generating grammatically consistent text using relatively small recurrent models trained on a single novel.

## 8. Repository Contents

- `ICT4403_Assignment2_Text_Generation.ipynb` — implementation and experiments.
- `model_comparison_results.csv` — quantitative model comparison.
- `training_validation_loss.png` — training and validation loss curves.
- `generated_text_samples.csv` — generated samples at different temperatures.
- `qualitative_generation_metrics.csv` — lexical diversity and repetition indicators.
- `the_time_machine.txt` — downloaded dataset.

## 9. Requirements and Execution

### Dependencies

- Python 3
- TensorFlow
- NumPy
- pandas
- Matplotlib

Install the dependencies with:

```bash
pip install tensorflow numpy pandas matplotlib
```

### Execution steps

1. Open the Jupyter notebook in Google Colab or Jupyter.
2. Run the cells sequentially.
3. Allow the dataset to download and all four models to train.
4. Inspect the comparison table, loss plot, generated text, and qualitative metrics.
5. Use the saved CSV files to reproduce the reported analysis.

**Note:** Training time and results can vary with the runtime environment and configuration.
