# Project Overview

This repository collects a series of text mining and NLP experiments into a single portfolio-style project.

## Workflows

| Notebook | Focus | Main Methods |
|---|---|---|
| `01_nlp_toolkit_comparison.ipynb` | NLP preprocessing and linguistic analysis | NLTK, spaCy |
| `02_sentiment_analysis_pipeline.ipynb` | Sentiment analysis and error analysis | VADER, Naive Bayes, TF-IDF, bag-of-words |
| `03_named_entity_recognition_classification.ipynb` | Named entity recognition | CoNLL features, SVM, Word2Vec embeddings |
| `04_topic_classification_transformers.ipynb` | Topic classification | RoBERTa, Naive Bayes, 20 Newsgroups |

## Repository Design

The notebooks are kept in `notebooks/` for exploration and presentation. Script exports are kept in `src/` for easier code review and reuse. Large datasets and pretrained models are documented but excluded from version control.
