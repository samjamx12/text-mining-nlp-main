"""
NLP Toolkit Comparison

Script export of the cleaned notebook workflow.
Run sections independently as needed, because some workflows require external datasets or pretrained models.
"""


# %% # NLP Toolkit Comparison


# %% ## Tip: how to read a file from disk


# %% Cell 2

from pathlib import Path



# %% Cell 3

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
path_to_file = PROJECT_ROOT / "data" / "raw" / "apple_samsung_sample.txt"
print(path_to_file)
print("does path exist? ->", path_to_file.exists())



# %% If the output from the code cell above states that **does path exist? -> False**, please check that 


# %% Cell 5

with open(path_to_file) as infile:
    text = infile.read()

print('number of characters', len(text))



# %% Cell 6

import nltk
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize



# %% Cell 7

sentences_nltk = sent_tokenize(text)



# %% Cell 8

tokens_per_sentence = []
for sentence_nltk in sentences_nltk:
    sent_tokens = word_tokenize(sentence_nltk)
    tokens_per_sentence.append(sent_tokens)



# %% We will use lists to keep track of the output of the NLP tasks. We can hence inspect the output for 


# %% Cell 10

sent_id = 1
print('SENTENCE', sentences_nltk[sent_id])
print('TOKENS', tokens_per_sentence[sent_id])



# %% Cell 11

pos_tags_per_sentence = []
for tokens in tokens_per_sentence:
    pos_tokens = nltk.pos_tag(tokens)
    pos_tags_per_sentence.append(pos_tokens)
    print(pos_tokens)



# %% Cell 12

print(pos_tags_per_sentence)



# %% Cell 13

ner_tags_per_sentence = []
for sentence in pos_tags_per_sentence:
    ner_tags = nltk.chunk.ne_chunk(sentence)
    ner_tags_per_sentence.append(ner_tags)



# %% Cell 14

print(ner_tags_per_sentence)



# %% Cell 15

constituent_parser = nltk.RegexpParser('''
NP: {<DT>? <JJ>* <NN>*} # NP
P: {<IN>}           # Preposition
V: {<V.*>}          # Verb
PP: {<P> <NP>}      # PP -> P NP
VP: {<V> <NP|PP>*}  # VP -> V (NP|PP)*''')



# %% Cell 16

constituency_output_per_sentence = []
for sentence in ner_tags_per_sentence:
    const = constituent_parser.parse(sentence)
    constituency_output_per_sentence.append(const)



# %% Cell 17

print(constituency_output_per_sentence)



# %% Augment the RegexpParser so that it also detects Named Entity Phrases (NEP), e.g., that it detects *


# %% Cell 19

constituent_parser_v2 = nltk.RegexpParser('''
NP: {<DT>? <JJ>* <NN>} # NP
P: {<IN>}           # Preposition
V: {<V.>}          # Verb
PP: {<P> <NP>}      # PP -> P NP
VP: {<V> <NP|PP>}  # VP -> V (NP|PP)
NEP: {<ORGANIZATION|PERSON>* <NNP>* <NNP>+}  # NEP -> (ORG|PERSON)* NNP* NNP+''')



# %% Cell 20

constituency_v2_output_per_sentence = []
for sentence in ner_tags_per_sentence:
    const = constituent_parser_v2.parse(sentence)
    constituency_v2_output_per_sentence.append(const)



# %% Cell 21

print(constituency_v2_output_per_sentence)



# %% Cell 22

import spacy
nlp = spacy.load('en_core_web_sm')



# %% ### a. Part of speech tagging


# %% Cell 24

doc = nlp(text)
sents = list(doc.sents)
third_sentence = sents[2]
for token in third_sentence:
    print(token.text, token.pos_, token.tag_)



# %% ### b. Named Enitity Recognition


# %% Cell 26

doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_)



# %% ### c. Constitunecy parsing


# %% Cell 28

from spacy import displacy



# %% Cell 29

doc = nlp(text)
displacy.render(doc, jupyter=True, style='dep')



# %% small tip: You can use **sents = list(doc.sents)** to be able to use the index to access a sentence 


# %% Cell 31

# Sentences one through five for NLTK
for sentence in sentences_nltk[:5]:
    tokens = word_tokenize(sentence)
    pos_tokens = nltk.pos_tag(tokens)
    pos_tags_per_sentence.append(pos_tokens)
    print(pos_tokens)
    print("-" * 50)



# %% Cell 32

# Sentences one through five for spaCy
for sent in list(doc.sents)[:5]:
    for token in sent:
        print(token.text, token.pos_, token.tag_)
    print('-' * 50)



# %% NLTK tends to split the sentences in connected blocks and when part of speech tagging is being opera


# %% Cell 34

# Load spaCy English model
nlp = spacy.load('en_core_web_sm')

# NLTK NER
nltk_pos_tagged = nltk.pos_tag(word_tokenize(text))
nltk_ner = nltk.chunk.ne_chunk(nltk_pos_tagged)

# spaCy NER
doc = nlp(text)

# NLTK NER Output
print("NLTK NER:")
print(nltk_ner)

# spaCy NER Output
print("\nspaCy NER:")
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")



# %% When observing both operations, the spaCy named entity recognition showcases a more accurate entity 


# %% Cell 36

# NLTK last sentence constituency parsing
constituency_output_per_sentence[-1].draw()



# %% Cell 37

# spaCy last sentence dependency parsing
last_sentence = list(doc.sents)[-1]
displacy.render(last_sentence, style='dep',
                jupyter=True)



# %% The similarity between the two is they are both have a tree sturcture however visualized differently
