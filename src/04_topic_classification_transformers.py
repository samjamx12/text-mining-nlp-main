"""
Topic Classification with Transformers

Script export of the cleaned notebook workflow.
Run sections independently as needed, because some workflows require external datasets or pretrained models.
"""


# %% # Topic Classification with Transformers


# %% Cell 1

# Import libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import classification_report
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import matplotlib.pyplot as plt
import seaborn as sn



# %% ### Getting and splitting the data


# %% Cell 3

from sklearn.datasets import fetch_20newsgroups

# load only a sub-selection of the categories (4 in our case)
categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'sci.space']

# remove the headers, footers and quotes (to avoid overfitting)
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=categories, random_state=42)
newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=categories, random_state=42)



# %% Cell 4

train = pd.DataFrame({'text': newsgroups_train.data, 'labels': newsgroups_train.target})
test = pd.DataFrame({'text': newsgroups_test.data, 'labels': newsgroups_test.target})



# %% Cell 5

from sklearn.model_selection import train_test_split

train, dev = train_test_split(train, test_size=0.1, random_state=0,
                               stratify=train[['labels']])



# %% ## **roBERTa model**


# %% ### Defining Model


# %% Cell 8

# Model configuration # https://simpletransformers.ai/docs/usage/#configuring-a-simple-transformers-model
model_args = ClassificationArgs()

model_args.overwrite_output_dir=True # overwrite existing saved models in the same directory
model_args.evaluate_during_training=True # to perform evaluation while training the model
# (eval data should be passed to the training method)

model_args.num_train_epochs=10 # number of epochs
model_args.train_batch_size=32 # batch size
model_args.learning_rate=4e-6 # learning rate
model_args.max_seq_length=256 # maximum sequence length
# Note! Increasing max_seq_len may provide better performance, but training time will increase.
# For educational purposes, we set max_seq_len to 256.

# Early stopping to combat overfitting: https://simpletransformers.ai/docs/tips-and-tricks/#using-early-stopping
model_args.use_early_stopping=True
model_args.early_stopping_delta=0.01 # "The improvement over best_eval_loss necessary to count as a better checkpoint"
model_args.early_stopping_metric='eval_loss'
model_args.early_stopping_metric_minimize=True
model_args.early_stopping_patience=2
model_args.evaluate_during_training_steps=32 # how often you want to run validation in terms of training steps (or batches)



# %% Cell 9

import torch

use_cuda = torch.cuda.is_available()
model = ClassificationModel("roberta", "roberta-base", num_labels=4, args=model_args, use_cuda=use_cuda)



# %% ### Training model


# %% Cell 11

_, history = model.train_model(train, eval_df=dev)



# %% Cell 12

# Training and evaluation loss
train_loss = history['train_loss']
eval_loss = history['eval_loss']
plt.plot(train_loss, label='Training loss')
plt.plot(eval_loss, label='Evaluation loss')
plt.title('Training and evaluation loss')
plt.legend()



# %% ### Evaluating model


# %% Cell 14

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(dev)
result



# %% Cell 15

predicted, probabilities = model.predict(test.text.to_list())
test['predicted'] = predicted



# %% Cell 16

print(classification_report(test['labels'], test['predicted']))



# %% ---


# %% ## **SVM/Naive Bayes model**


# %% Cell 19

### Implement an SVM/Naive Bayes model for topic classification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score

# Pipeline for Naive Bayes
nb_pipeline = make_pipeline(TfidfVectorizer(stop_words='english', max_features=10000), MultinomialNB())

# Fit the model
nb_pipeline.fit(train['text'], train['labels'])

# Evaluate the model on the development set
dev_preds_nb = nb_pipeline.predict(dev['text'])
nb_accuracy = accuracy_score(dev['labels'], dev_preds_nb)

print("Naive Bayes Classification Report:")
print(classification_report(dev['labels'], dev_preds_nb))



# %% Cell 20

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt

# Generating predictions
dev_preds_nb = nb_pipeline.predict(dev['text'])

# Confusion matrix
cm = confusion_matrix(dev['labels'], dev_preds_nb)
labels = ['alt.atheism', 'comp.graphics', 'sci.med', 'sci.space']

# Plot confusion matrix plot
fig, ax = plt.subplots(figsize=(5, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax)
plt.title('Naive Bayes Confusion Matrix')
plt.show()

# F1 scores calculated and plotted in a bar chart
f1_scores = f1_score(dev['labels'], dev_preds_nb, average=None)
fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(labels, f1_scores, )
ax.set_xlabel('Category')
ax.set_ylabel('F1 Score')
ax.set_title('F1 Scores per category for Naive Bayes')
ax.tick_params(axis='x', labelrotation=45)
plt.tight_layout()
plt.show()



# %% ## **Comparing the models**


# %% Report for roBERTa:
