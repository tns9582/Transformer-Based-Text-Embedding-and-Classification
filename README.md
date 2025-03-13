# Semantic Intelligence: Transformer-Based Text Embedding and Classification

This project implements a custom transformer-based model to learn meaningful embeddings of news articles and perform multi-class classification using the AG News dataset. It demonstrates how deep learning architectures, particularly attention and positional encoding, can be used to capture semantic relationships in text, evaluate classification accuracy, and explore interpretable and reversible embeddings.

## Project Overview

The goal of this project is to analyze over 100,000 news headlines from the AG News dataset and classify them into four categories: **World**, **Sports**, **Business**, and **Sci/Tech**. Instead of using pre-built models, we build our own transformer-inspired architecture using **TensorFlow/Keras** from the ground up.

Key components include:
- Word, position, and contextual embeddings
- Self-attention mechanism
- Transformer layers with residual connections
- Softmax-based text classification
- Interpretable binary embeddings and reverse mapping
- Sentence similarity via cosine distance and vector arithmetic


## Tools & Technologies

- **Language**: Python  
- **Libraries**: TensorFlow, Keras, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn  
- **Dataset**: [AG News Corpus](https://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)  
- **Modeling**: Custom Transformer-based classifier  
- **Evaluation Metrics**: Confusion Matrix, Precision, Recall, Cosine Similarity, Hamming Distance


## Key Features

- Implements core components of a transformer (attention, position, feedforward)
- Demonstrates the power of contextual embeddings in capturing sentence meaning
- Shows how vector operations can model semantic relationships
- Introduces binary interpretable embeddings and reverse-lookup mechanisms


## Results & Insights

- Achieved strong classification performance across all four news categories
- Demonstrated semantic analogies using vector arithmetic
- Verified sentence similarity via cosine scoring
- Reconstructed input tokens from binary embeddings using Hamming distance

## Future Improvements

- Incorporate pre-trained embeddings (e.g., GloVe, BERT) for comparison  
- Expand to longer text inputs with positional encoding optimization  
- Experiment with fine-tuning on downstream NLP tasks  
