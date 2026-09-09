"""
Sentiment Analysis Pipeline

Script export of the cleaned notebook workflow.
Run sections independently as needed, because some workflows require external datasets or pretrained models.
"""


# %% # Sentiment Analysis Pipeline


# %% ###   Collecting 50 tweets for evaluation


# %% You can load your tweets with human annotation in the following way.


# %% Cell 3

import json



# %% Cell 4

from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
tweets_path = PROJECT_ROOT / "data" / "raw" / "my_tweets.json"
my_tweets = json.load(open(tweets_path, encoding="utf-8"))



# %% Cell 5

for id_, tweet_info in my_tweets.items():
    print(id_, tweet_info)
    break



# %% ### Findings


# %% Cell 7

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
nlp = spacy.load('en_core_web_sm')
vader_model = SentimentIntensityAnalyzer()

def run_vader(textual_unit,
              lemmatize=False,
              parts_of_speech_to_consider=None,
              verbose=0):
    """
    Run VADER on a sentence from spacy

    :param str textual unit: a textual unit, e.g., sentence, sentences (one string)
    (by looping over doc.sents)
    :param bool lemmatize: If True, provide lemmas to VADER instead of words
    :param set parts_of_speech_to_consider:
    -None or empty set: all parts of speech are provided
    -non-empty set: only these parts of speech are considered.
    :param int verbose: if set to 1, information is printed
    about input and output

    :rtype: dict
    :return: vader output dict
    """
    doc = nlp(textual_unit)

    input_to_vader = []

    for sent in doc.sents:
        for token in sent:

            to_add = token.text

            if lemmatize:
                to_add = token.lemma_

                if to_add == '-PRON-':
                    to_add = token.text

            if parts_of_speech_to_consider:
                if token.pos_ in parts_of_speech_to_consider:
                    input_to_vader.append(to_add)
            else:
                input_to_vader.append(to_add)

    scores = vader_model.polarity_scores(' '.join(input_to_vader))

    if verbose >= 1:
        print()
        print('INPUT SENTENCE', sent)
        print('INPUT TO VADER', input_to_vader)
        print('VADER OUTPUT', scores)

    return scores



# %% Cell 8

def vader_output_to_label(vader_output):
    """
    map vader output e.g.,
    {'neg': 0.0, 'neu': 0.0, 'pos': 1.0, 'compound': 0.4215}
    to one of the following values:
    a) positive float -> 'positive'
    b) 0.0 -> 'neutral'
    c) negative float -> 'negative'

    :param dict vader_output: output dict from vader

    :rtype: str
    :return: 'negative' | 'neutral' | 'positive'
    """
    compound = vader_output['compound']

    if compound < 0:
        return 'negative'
    elif compound == 0.0:
        return 'neutral'
    elif compound > 0.0:
        return 'positive'

assert vader_output_to_label( {'neg': 0.0, 'neu': 0.0, 'pos': 1.0, 'compound': 0.0}) == 'neutral'
assert vader_output_to_label( {'neg': 0.0, 'neu': 0.0, 'pos': 1.0, 'compound': 0.01}) == 'positive'
assert vader_output_to_label( {'neg': 0.0, 'neu': 0.0, 'pos': 1.0, 'compound': -0.01}) == 'negative'



# %% Cell 9

tweets = []
all_vader_output = []
gold = []

# settings (to change for different experiments)
to_lemmatize = True
pos = set()

for id_, tweet_info in my_tweets.items():
    the_tweet = tweet_info['text_of_tweet']
    vader_output = run_vader(the_tweet) # run vader
    vader_label = vader_output_to_label(vader_output) # convert vader output to category

    tweets.append(the_tweet)
    all_vader_output.append(vader_label)
    gold.append(tweet_info['sentiment_label'])

# use scikit-learn's classification report
for i, tweet in enumerate(tweets):
    print()
    print(tweet)
    print("Vader:", all_vader_output[i])
    print("Annotation:", gold[i])
    



# %% #### Findings


# %% Cell 11

# Load airline tweet files from the project data directory

import os
import pathlib
from sklearn.datasets import load_files
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

PROJECT_ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd()
airline_tweets_folder = PROJECT_ROOT / "data" / "raw" / "airlinetweets"
airline_tweets_train = load_files(str(airline_tweets_folder))



# %% Cell 12

import numpy as np
def classification_report_own(negative_tweets_annotations, positive_tweets_annotations, neutral_tweets_annotations):
    true_negative = [1 for annot in negative_tweets_annotations]
    true_positive = [1 for annot in positive_tweets_annotations]
    true_neutral = [1 for annot in neutral_tweets_annotations]

    predicted_neg = [1 if annot == 'negative' else 0 for annot in negative_tweets_annotations]
    predicted_pos = [1 if annot == 'positive' else 0 for annot in positive_tweets_annotations]
    predicted_neutral = [1 if annot == 'neutral' else 0 for annot in neutral_tweets_annotations]

    true_positive_count_neg = sum([1 for true_label, predicted_label in zip(true_negative, predicted_neg) if true_label == 1 and predicted_label == 1])
    true_negative_count_neg = sum([1 for true_label, predicted_label in zip(true_negative, predicted_neg) if true_label != 1 and predicted_label != 1])
    false_positive_count_neg = sum([1 for true_label, predicted_label in zip(true_negative, predicted_neg) if true_label != 1 and predicted_label == 1])
    false_negative_count_neg = sum([1 for true_label, predicted_label in zip(true_negative, predicted_neg) if true_label == 1 and predicted_label != 1])

    true_positive_count_pos = sum([1 for true_label, predicted_label in zip(true_positive, predicted_pos) if true_label == 1 and predicted_label == 1])
    true_negative_count_pos = sum([1 for true_label, predicted_label in zip(true_positive, predicted_pos) if true_label != 1 and predicted_label != 1])
    false_positive_count_pos = sum([1 for true_label, predicted_label in zip(true_positive, predicted_pos) if true_label != 1 and predicted_label == 1])
    false_negative_count_pos = sum([1 for true_label, predicted_label in zip(true_positive, predicted_pos) if true_label == 1 and predicted_label != 1])

    true_positive_count_neutral = sum([1 for true_label, predicted_label in zip(true_neutral, predicted_neutral) if true_label == 1 and predicted_label == 1])
    true_negative_count_neutral = sum([1 for true_label, predicted_label in zip(true_neutral, predicted_neutral) if true_label != 1 and predicted_label != 1])
    false_positive_count_neutral = sum([1 for true_label, predicted_label in zip(true_neutral, predicted_neutral) if true_label != 1 and predicted_label == 1])
    false_negative_count_neutral = sum([1 for true_label, predicted_label in zip(true_neutral, predicted_neutral) if true_label == 1 and predicted_label != 1])

    # Precision
    neg_precision = precision_score(true_negative, predicted_neg)
    pos_precision = precision_score(true_positive, predicted_pos)
    neutral_precision = precision_score(true_neutral, predicted_neutral)

    # Recall
    neg_recall = recall_score(true_negative, predicted_neg)
    pos_recall = recall_score(true_positive, predicted_pos)
    neutral_recall = recall_score(true_neutral, predicted_neutral)

    # F1 score
    neg_f1 = f1_score(true_negative, predicted_neg)
    pos_f1 = f1_score(true_positive, predicted_pos)
    neutral_f1 = f1_score(true_neutral, predicted_neutral)

    # Macro averages
    prec_macro = (neg_precision + pos_precision + neutral_precision) / 3
    rec_macro = (neg_recall + pos_recall + neutral_recall) / 3
    f1_macro = (neg_f1 + pos_f1 + neutral_f1) / 3

    # Micro averages
    # Calculate micro-average precision
    total_true_positive = true_positive_count_neg + true_positive_count_pos + true_positive_count_neutral
    total_false_positive = false_positive_count_neg + false_positive_count_pos + false_positive_count_neutral
    micro_precision = total_true_positive / (total_true_positive + total_false_positive)

    # Calculate micro-average recall
    total_false_negative = false_negative_count_neg + false_negative_count_pos + false_negative_count_neutral
    micro_recall = total_true_positive / (total_true_positive + total_false_negative)

    # Calculate micro-average F1-score
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)

    print(f"Negative precision: {neg_precision:.3f}  Neutral Precision: {neutral_precision:.3f}  Positive precision: {pos_precision:.3f}")
    print(f"Negative recall: {neg_recall:.3f}     Neutral Recall: {neutral_recall:.3f}     Positive recall: {pos_recall:.3f}")
    print(f"Negative F1: {neg_f1:.3f}         Neutral F1: {neutral_f1:.3f}         Positive F1: {pos_f1:.3f}")

    print()
    print("---------------Macro Averages---------------")
    print(f"Precision: {prec_macro:.3f}; Recall: {rec_macro:.3f}; F1: {f1_macro:.3f}")
    print()
    print("---------------Micro Averages---------------")
    print(f"Precision: {micro_precision:.3f}; Recall: {micro_recall:.3f}; F1: {micro_f1:.3f}")



# %% ### Vader analysis tweets as is


# %% Cell 14

# Tweets as is
tweets_annotations = []
for i, textb in enumerate(airline_tweets_train.data):
    text = textb.decode('utf-8')
    vader_output = run_vader(text)
    vader_label = vader_output_to_label(vader_output)
    tweets_annotations.append(vader_label)

neg_tweets = [tweets_annotations[i] for i in range(len(tweets_annotations)) if airline_tweets_train.target[i] == 0]
neutral_tweets = [tweets_annotations[i] for i in range(len(tweets_annotations)) if airline_tweets_train.target[i] == 1]
pos_tweets = [tweets_annotations[i] for i in range(len(tweets_annotations)) if airline_tweets_train.target[i] == 2]



# %% Cell 15

classification_report_own(neg_tweets, pos_tweets, neutral_tweets)
y_pred = [0 if ann == 'negative' else(1 if ann == 'neutral' else 2) for ann in tweets_annotations]
print('\n' + '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' + '\n')
print(classification_report(airline_tweets_train.target, y_pred))



# %% ### Vader analysis tweets lemmatized


# %% Cell 17

# Tweets lemmatized
tweets_annotations = []
for i, textb in enumerate(airline_tweets_train.data):
    text = textb.decode('utf-8')
    vader_output = run_vader(text, lemmatize=True)
    vader_label = vader_output_to_label(vader_output)
    tweets_annotations.append(vader_label)

neg_tweets = [tweets_annotations[i] for i in range(len(tweets_annotations)) if airline_tweets_train.target[i] == 0]
neutral_tweets = [tweets_annotations[i] for i in range(len(tweets_annotations)) if airline_tweets_train.target[i] == 1]
pos_tweets = [tweets_annotations[i] for i in range(len(tweets_annotations)) if airline_tweets_train.target[i] == 2]



# %% Cell 18

classification_report_own(neg_tweets, pos_tweets, neutral_tweets)
y_pred = [0 if ann == 'negative' else(1 if ann == 'neutral' else 2) for ann in tweets_annotations]
print('\n' + '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' + '\n')
print(classification_report(airline_tweets_train.target, y_pred))



# %% ### Vader analysis tweets only adjectives


# %% Cell 20

# Tweets using only adjectives
tweets_annotations = []
for i, textb in enumerate(airline_tweets_train.data):
    text = textb.decode('utf-8')
    vader_output = run_vader(text, parts_of_speech_to_consider={'ADJ'})
    vader_label = vader_output_to_label(vader_output)
    tweets_annotations.append(vader_label)

neg_tweets = [tweets_annotations[i] for i in range(len(tweets_annotations)) if airline_tweets_train.target[i] == 0]
neutral_tweets = [tweets_annotations[i] for i in range(len(tweets_annotations)) if airline_tweets_train.target[i] == 1]
pos_tweets = [tweets_annotations[i] for i in range(len(tweets_annotations)) if airline_tweets_train.target[i] == 2]



# %% Cell 21

classification_report_own(neg_tweets, pos_tweets, neutral_tweets)
y_pred = [0 if ann == 'negative' else(1 if ann == 'neutral' else 2) for ann in tweets_annotations]
print('\n' + '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' + '\n')
print(classification_report(airline_tweets_train.target, y_pred))



# %% ### Vader analysis tweets lemmatized & only adjectives


# %% Cell 23

# Tweets lemmatized and using only adjectives
tweets_annotations = []
for i, textb in enumerate(airline_tweets_train.data):
    text = textb.decode('utf-8')
    vader_output = run_vader(text, lemmatize=True, parts_of_speech_to_consider={'ADJ'})
    vader_label = vader_output_to_label(vader_output)
    tweets_annotations.append(vader_label)

neg_tweets = [tweets_annotations[i] for i in range(len(tweets_annotations)) if airline_tweets_train.target[i] == 0]
neutral_tweets = [tweets_annotations[i] for i in range(len(tweets_annotations)) if airline_tweets_train.target[i] == 1]
pos_tweets = [tweets_annotations[i] for i in range(len(tweets_annotations)) if airline_tweets_train.target[i] == 2]



# %% Cell 24

classification_report_own(neg_tweets, pos_tweets, neutral_tweets)
y_pred = [0 if ann == 'negative' else(1 if ann == 'neutral' else 2) for ann in tweets_annotations]
print('\n' + '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' + '\n')
print(classification_report(airline_tweets_train.target, y_pred))



# %% ### Vader analysis tweets only nouns


# %% Cell 26

# Tweets using only nouns
tweets_annotations = []
for i, textb in enumerate(airline_tweets_train.data):
    text = textb.decode('utf-8')
    vader_output = run_vader(text, parts_of_speech_to_consider={'NOUN'})
    vader_label = vader_output_to_label(vader_output)
    tweets_annotations.append(vader_label)

neg_tweets = [tweets_annotations[i] for i in range(len(tweets_annotations)) if airline_tweets_train.target[i] == 0]
neutral_tweets = [tweets_annotations[i] for i in range(len(tweets_annotations)) if airline_tweets_train.target[i] == 1]
pos_tweets = [tweets_annotations[i] for i in range(len(tweets_annotations)) if airline_tweets_train.target[i] == 2]



# %% Cell 27

classification_report_own(neg_tweets, pos_tweets, neutral_tweets)
y_pred = [0 if ann == 'negative' else(1 if ann == 'neutral' else 2) for ann in tweets_annotations]
print('\n' + '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' + '\n')
print(classification_report(airline_tweets_train.target, y_pred))



# %% ### Vader analysis tweets lemmatized & only nouns


# %% Cell 29

# Tweets lemmatized and using only nouns
tweets_annotations = []
for i, textb in enumerate(airline_tweets_train.data):
    text = textb.decode('utf-8')
    vader_output = run_vader(text, lemmatize=True, parts_of_speech_to_consider={'NOUN'})
    vader_label = vader_output_to_label(vader_output)
    tweets_annotations.append(vader_label)

neg_tweets = [tweets_annotations[i] for i in range(len(tweets_annotations)) if airline_tweets_train.target[i] == 0]
neutral_tweets = [tweets_annotations[i] for i in range(len(tweets_annotations)) if airline_tweets_train.target[i] == 1]
pos_tweets = [tweets_annotations[i] for i in range(len(tweets_annotations)) if airline_tweets_train.target[i] == 2] 



# %% Cell 30

classification_report_own(neg_tweets, pos_tweets, neutral_tweets)
y_pred = [0 if ann == 'negative' else(1 if ann == 'neutral' else 2) for ann in tweets_annotations]
print('\n' + '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' + '\n')
print(classification_report(airline_tweets_train.target, y_pred))



# %% ### Vader analysis tweets only verbs


# %% Cell 32

# Tweets using only verbs
tweets_annotations = []
for i, textb in enumerate(airline_tweets_train.data):
    text = textb.decode('utf-8')
    vader_output = run_vader(text, parts_of_speech_to_consider={'VERB'})
    vader_label = vader_output_to_label(vader_output)
    tweets_annotations.append(vader_label)

neg_tweets = [tweets_annotations[i] for i in range(len(tweets_annotations)) if airline_tweets_train.target[i] == 0]
neutral_tweets = [tweets_annotations[i] for i in range(len(tweets_annotations)) if airline_tweets_train.target[i] == 1]
pos_tweets = [tweets_annotations[i] for i in range(len(tweets_annotations)) if airline_tweets_train.target[i] == 2]



# %% Cell 33

classification_report_own(neg_tweets, pos_tweets, neutral_tweets)
y_pred = [0 if ann == 'negative' else(1 if ann == 'neutral' else 2) for ann in tweets_annotations]
print('\n' + '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' + '\n')
print(classification_report(airline_tweets_train.target, y_pred))



# %% ### Vader analysis tweets lemmatized & only verbs


# %% Cell 35

# Tweets lemmatized and using only verbs
tweets_annotations = []
for i, textb in enumerate(airline_tweets_train.data):
    text = textb.decode('utf-8')
    vader_output = run_vader(text, lemmatize=True, parts_of_speech_to_consider={'VERB'})
    vader_label = vader_output_to_label(vader_output)
    tweets_annotations.append(vader_label)

neg_tweets = [tweets_annotations[i] for i in range(len(tweets_annotations)) if airline_tweets_train.target[i] == 0]
neutral_tweets = [tweets_annotations[i] for i in range(len(tweets_annotations)) if airline_tweets_train.target[i] == 1]
pos_tweets = [tweets_annotations[i] for i in range(len(tweets_annotations)) if airline_tweets_train.target[i] == 2] 



# %% Cell 36

classification_report_own(neg_tweets, pos_tweets, neutral_tweets)
y_pred = [0 if ann == 'negative' else(1 if ann == 'neutral' else 2) for ann in tweets_annotations]
print('\n' + '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' + '\n')
print(classification_report(airline_tweets_train.target, y_pred))



# %% ### Model Comparison Notes


# %% Cell 38

"""
This section compares several vectorization settings in a compact experiment loop.
"""

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords


tfidf_transformer = TfidfTransformer()
airline_vec = CountVectorizer(min_df=2, # If a token appears fewer times than this, across all documents, it will be ignored
                             tokenizer=nltk.word_tokenize, # we use the nltk tokenizer
                             stop_words=stopwords.words('english')) # stopwords are removed

airline_counts = airline_vec.fit_transform(airline_tweets_train.data)


airline_tfidf = tfidf_transformer.fit_transform(airline_counts)

x_train, x_test, y_train, y_test = train_test_split(airline_tfidf,
                                                   airline_tweets_train.target,
                                                   test_size=0.2)

base_model = MultinomialNB().fit(x_train, y_train)

y_pred = base_model.predict(x_test)

print(classification_report(y_test, y_pred))



# %% #### Using Bag Of Words representation instead of TF-IDF


# %% Cell 40

x_train, x_test, y_train, y_test = train_test_split(airline_counts,
                                                   airline_tweets_train.target,
                                                   test_size=0.2)

base_model = MultinomialNB().fit(x_train, y_train)

y_pred = base_model.predict(x_test)

print(classification_report(y_test, y_pred))



# %% #### Using min_df = 5


# %% Cell 42

airline_vec = CountVectorizer(min_df=5, # If a token appears fewer times than this, across all documents, it will be ignored
                             tokenizer=nltk.word_tokenize, # we use the nltk tokenizer
                             stop_words=stopwords.words('english')) # stopwords are removed

airline_counts = airline_vec.fit_transform(airline_tweets_train.data)


airline_tfidf = tfidf_transformer.fit_transform(airline_counts)

x_train, x_test, y_train, y_test = train_test_split(airline_tfidf,
                                                   airline_tweets_train.target,
                                                   test_size=0.2)

base_model = MultinomialNB().fit(x_train, y_train)

y_pred = base_model.predict(x_test)

print(classification_report(y_test, y_pred))



# %% #### Using min_df = 10


# %% Cell 44

airline_vec = CountVectorizer(min_df=10, # If a token appears fewer times than this, across all documents, it will be ignored
                             tokenizer=nltk.word_tokenize, # we use the nltk tokenizer
                             stop_words=stopwords.words('english')) # stopwords are removed

airline_counts = airline_vec.fit_transform(airline_tweets_train.data)


airline_tfidf = tfidf_transformer.fit_transform(airline_counts)

x_train, x_test, y_train, y_test = train_test_split(airline_tfidf,
                                                   airline_tweets_train.target,
                                                   test_size=0.2)

base_model = MultinomialNB().fit(x_train, y_train)

y_pred = base_model.predict(x_test)

print(classification_report(y_test, y_pred))



# %% Cell 45

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import nltk

airline_vec = CountVectorizer(min_df=2, # If a token appears fewer times than this, across all documents, it will be ignored
                             tokenizer=nltk.word_tokenize, # we use the nltk tokenizer
                             stop_words=stopwords.words('english')) # stopwords are removed


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    airline_tweets_train.data,
    airline_tweets_train.target,
    test_size=0.20
)

vectorizers = {
    'TF-IDF': TfidfVectorizer,
    'Bag of Words': CountVectorizer
}

min_dfs = [2, 5, 10]

# Iterate over different settings
for vectorizer_name, vectorizer_class in vectorizers.items():
    for min_df in min_dfs:
        # Define vectorizer with current settings
        vectorizer = vectorizer_class(min_df=min_df, tokenizer=nltk.word_tokenize, stop_words='english')

        # Vectorize training and test data
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)

        # Train Naive Bayes classifier
        clf = MultinomialNB()
        clf.fit(X_train_vectorized, y_train)

        # Predict on test data
        y_pred = clf.predict(X_test_vectorized)

        # Evaluate performance
        print(f"Vectorizer: {vectorizer_name}, min_df: {min_df}")
        print(classification_report(y_test, y_pred))
        print("-------------------------------------------------------")



# %% ### Feature Interpretation


# %% Cell 47

from sklearn.feature_extraction.text import CountVectorizer

def important_features_per_class(vectorizer,classifier,n=80):
    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names_out()
    topn_class1 = sorted(zip(classifier.feature_count_[0], feature_names),reverse=True)[:n]
    topn_class2 = sorted(zip(classifier.feature_count_[1], feature_names),reverse=True)[:n]
    topn_class3 = sorted(zip(classifier.feature_count_[2], feature_names),reverse=True)[:n]
    print("Important words in negative documents")
    for coef, feat in topn_class1:
        print(class_labels[0], coef, feat)
    print("-----------------------------------------")
    print("Important words in neutral documents")
    for coef, feat in topn_class2:
        print(class_labels[1], coef, feat)
    print("-----------------------------------------")
    print("Important words in positive documents")
    for coef, feat in topn_class3:
        print(class_labels[2], coef, feat)

# Example call:
#important_features_per_class(airline_vec, clf)

# Define CountVectorizer with min_df=2
airline_vec = CountVectorizer(min_df=2, # If a token appears fewer times than this, across all documents, it will be ignored
                             tokenizer=nltk.word_tokenize, # we use the nltk tokenizer
                             stop_words=stopwords.words('english')) # stopwords are removed

airline_counts = airline_vec.fit_transform(airline_tweets_train.data)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    airline_counts,
    airline_tweets_train.target,
    test_size=0.20
)
# Train Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

important_features_per_class(airline_vec, clf)

