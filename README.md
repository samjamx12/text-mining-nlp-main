# Text Mining and NLP Portfolio

A collection of NLP and text mining projects covering toolkit comparison, sentiment analysis, named entity recognition, and topic classification.

## Projects Included

### 1. NLP Toolkit Comparison
Compares NLTK and spaCy for common linguistic processing tasks:
- Sentence splitting and tokenization
- POS tagging
- Named entity recognition
- Constituency-style chunking
- Dependency parsing

### 2. Sentiment Analysis Pipeline
Builds sentiment analysis workflows using:
- VADER rule-based sentiment scoring
- Preprocessing experiments with lemmatization and part-of-speech filtering
- Naive Bayes classification with bag-of-words and TF-IDF features
- Classification reports and feature inspection

### 3. Named Entity Recognition Classification
Explores supervised NER using:
- CoNLL-style token/POS features
- Dictionary vectorization
- Linear SVM classifiers
- Word2Vec embeddings
- Error analysis and class-level evaluation

### 4. Topic Classification with Transformers
Compares transformer and classical models on 20 Newsgroups:
- RoBERTa fine-tuning
- Naive Bayes baseline
- Precision, recall, F1-score, and confusion matrix evaluation

## Repository Structure

```text
.
├── notebooks/          # Cleaned Jupyter notebooks
├── src/                # Python script exports of the notebooks
├── data/
│   └── raw/            # Small samples and dataset instructions
├── models/             # External pretrained models, not committed
├── results/            # Summary of outputs and findings
├── reports/            # Space for generated reports/figures
├── requirements.txt
├── PROJECT_OVERVIEW.md
├── .gitignore
└── README.md
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt averaged_perceptron_tagger maxent_ne_chunker words vader_lexicon stopwords
```

## Running the Notebooks

Start Jupyter:

```bash
jupyter notebook
```

Open the notebooks in order:

```text
notebooks/01_nlp_toolkit_comparison.ipynb
notebooks/02_sentiment_analysis_pipeline.ipynb
notebooks/03_named_entity_recognition_classification.ipynb
notebooks/04_topic_classification_transformers.ipynb
```

## Data Setup

A small sample text file and sample `my_tweets.json` are included. Larger datasets are not included.

See `data/raw/README.md` for expected dataset locations and formats.

## Technologies Used

- Python
- Jupyter Notebook
- NLTK
- spaCy
- VADER
- scikit-learn
- Gensim
- PyTorch
- Transformers
- Simple Transformers
- Matplotlib

## What I Learned

This project demonstrates how different NLP approaches behave across the text mining pipeline: rule-based NLP, statistical machine learning, pretrained embeddings, and transformer-based classification. It also highlights the importance of preprocessing choices, label imbalance, evaluation metrics, and qualitative error analysis.

## Future Improvements

- Add reusable command-line scripts for each workflow
- Store generated plots in `reports/`
- Add automated tests for preprocessing functions
- Replace manual tweet annotations with a larger labeled dataset
- Compare more transformer models for topic classification
- Add model cards for saved classifiers
