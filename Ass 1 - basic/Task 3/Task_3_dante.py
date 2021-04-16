import nltk  
import numpy as np  
import random  
import string
import bs4 as bs  
import re 
import pandas as pd
import heapq

""" Setting up vectorization through bag of words method
Mostly build upon info from:
https://stackabuse.com/python-for-nlp-creating-bag-of-words-model-from-scratch/
"""

df = pd.read_csv("Task 3/Sms_new.csv")

for count, text in enumerate(df["text"]):
    text = str(text)
    text = text.lower()
    text = re.sub(r'\W',' ', text)
    text = re.sub(r'\s+',' ', text)
    df.loc[df.index[count], "text"] = text

# print(df.head())

wordfreq = {}
for text in df["text"]:
    tokens = nltk.word_tokenize(text)
    for token in tokens:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1

print(len(wordfreq))

most_freq = heapq.nlargest(500, wordfreq, key=wordfreq.get)

# print(most_freq)

sentence_vectors = []
for text in df["text"]:
    sentence_tokens = nltk.word_tokenize(text)
    sent_vec = []
    for token in most_freq:
        if token in sentence_tokens:
            sent_vec.append(1)
        else:
            sent_vec.append(0)
    sentence_vectors.append(sent_vec)

sentence_vectors = np.asarray(sentence_vectors)
# print(sentence_vectors)

label_vec = []
for label in df["label"]:
    if label == "ham":
        label_vec.append(0)
    else:
        label_vec.append(1)

df_wordvec = pd.DataFrame(sentence_vectors, columns = most_freq)

df_wordvec["Label"] = label_vec

print(df_wordvec.head())

# df_wordvec.to_csv("Ass 1 - basic/Task 3/sms_vec.csv")



