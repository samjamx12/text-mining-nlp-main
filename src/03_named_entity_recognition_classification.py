"""
Named Entity Recognition Classification

Script export of the cleaned notebook workflow.
Run sections independently as needed, because some workflows require external datasets or pretrained models.
"""


# %% # Named Entity Recognition Classification


# %% Cell 1

from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
from nltk.corpus.reader import ConllCorpusReader
train = ConllCorpusReader(str(PROJECT_ROOT / 'data' / 'raw' / 'CONLL2003'), 'train.txt', ['words', 'pos', 'ignore', 'chunk'])
training_features = []
training_gold_labels = []

for token, pos, ne_label in train.iob_words():
    a_dict = {
        'token': token,
        'pos': pos
    }
    training_features.append(a_dict)
    training_gold_labels.append(ne_label)
training_features[0:10]



# %% Cell 2

training_gold_labels[0:10]



# %% Cell 3

from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
test = ConllCorpusReader(str(PROJECT_ROOT / 'data' / 'raw' / 'CONLL2003'), 'test.txt', ['words', 'pos', 'ignore', 'chunk'])

test_features = []
test_gold_labels = []
for token, pos, ne_label in test.iob_words():
    a_dict = {
        'token': token,
        'pos': pos
    }
    test_features.append(a_dict)
    test_gold_labels.append(ne_label)
test_features[0:10]



# %% Cell 4

test_gold_labels[0:10]



# %% Cell 5

from collections import Counter
import matplotlib.pyplot as plt
print('train size:', len(training_features))
print('test size:', len(test_features))

train_count = Counter(training_gold_labels)
test_count = Counter(test_gold_labels)
classes = train_count.keys()

cumulative_freq = {class_label: train_count[class_label] + test_count[class_label] for class_label in classes}
sorted_classes = sorted(classes, key=lambda x: cumulative_freq[x], reverse=True)
sorted_classes = [c for c in sorted_classes if c != 'O']

train_counts_list = [train_count[class_label] for class_label in sorted_classes if class_label != 'O']
test_counts_list = [test_count[class_label] for class_label in sorted_classes if class_label != 'O']
#train_counts_list = [train_count[class_label] for class_label in sorted_classes]
#test_counts_list = [test_count[class_label] for class_label in sorted_classes]

plt.figure(figsize=(10, 6))
plt.bar(sorted_classes, train_counts_list, color='skyblue', label='Training Instances')
plt.bar(sorted_classes, test_counts_list, color='lightgreen', label='Test Instances')

plt.title('Frequency of Each Class')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.legend()
plt.show()



# %% How many instances are in train and test?


# %% Cell 7

from sklearn.feature_extraction import DictVectorizer



# %% Cell 8

vec = DictVectorizer()
the_array = training_features + test_features
vec_array = vec.fit_transform(the_array)



# %% Cell 9

from sklearn import svm



# %% Cell 10

lin_clf = svm.LinearSVC()



# %% Cell 11

from sklearn.metrics import classification_report

x_train = vec_array[:203621]
x_test = vec_array[203621:]

lin_clf.fit(x_train, training_gold_labels)
y_pred = lin_clf.predict(x_test)

print(classification_report(test_gold_labels, y_pred))



# %% *NERC labels the classifier performs well on:*


# %% Cell 13

from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
from gensim.models import KeyedVectors

# Load pre-trained Word2Vec embeddings
word2vec_model = KeyedVectors.load_word2vec_format(str(PROJECT_ROOT / 'models' / 'GoogleNews-vectors-negative300.bin'), binary=True)



# %% Cell 14

train_vectors=[]
train_labels=[]
for token, pos, ne_label in train.iob_words():

    if token!='' and token!='DOCSTART':
        if token in word2vec_model:
            vector=word2vec_model[token]
        else:
            vector=[0]*300
        train_vectors.append(vector)
        train_labels.append(ne_label)

test_vectors=[]
test_labels=[]
for token, pos, ne_label in test.iob_words():

    if token!='' and token!='DOCSTART':
        if token in word2vec_model:
            vector=word2vec_model[token]
        else:
            vector=[0]*300
        test_vectors.append(vector)
        test_labels.append(ne_label)



# %% Cell 15

emb_model = svm.LinearSVC()
emb_model.fit(train_vectors, train_labels)

y_pred = emb_model.predict(test_vectors)
print(classification_report(test_labels, y_pred))



# %% *Without word embeddings the classifier's performance on NERC labels:*


# %% Cell 17

import pandas as pd



# %% Cell 18

from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
try:
    kaggle_dataset = pd.read_csv(PROJECT_ROOT / 'data' / 'raw' / 'ner_dataset.csv', encoding='latin1')
except UnicodeDecodeError:
    print("There was an error decoding the CSV file. Please try a different encoding or handle the issue manually.")



# %% Cell 19

len(kaggle_dataset)



# %% Cell 20

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')

vec = DictVectorizer()
dict_data = kaggle_dataset[['Word', 'POS']].to_dict(orient='records')
vec_data = vec.fit_transform(dict_data)

df_train = vec_data[:838861]
df_test = vec_data[838861:]
#df_train = vec_data[:100000]
#df_test = vec_data[100000:120000]

df_train_imp = imputer.fit_transform(df_train)
df_test_imp = imputer.fit_transform(df_test)


y_train = kaggle_dataset[['Tag']][:838861]
y_test = kaggle_dataset[['Tag']][838861:]

print(df_train.shape, df_test.shape)



# %% Cell 21

clf = svm.LinearSVC()

clf.fit(df_train_imp, y_train)
y_pred = clf.predict(df_test_imp)

print(classification_report(y_test, y_pred))



# %% The SVM model demonstrated its capacity to accurately categorize cases by achieving a 95% accuracy o
