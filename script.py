# coding: utf-8

import pandas as pd
import numpy as np
import csv
import re
import sys
import os

lang1_data = str(sys.argv[1])
lang2_data = str(sys.argv[2])

test_data = str(sys.argv[3])

train_dataL1 = "samp"+lang1_data
train_dataL2 = "samp"+lang2_data

# Subsampling the two datasets (taking 20k from all three)
os.system("shuf -n 50000 " + str(lang1_data) + " > " + str(train_dataL1))
os.system("shuf -n 50000 " + str(lang2_data) + " > " + str(train_dataL2))


names = ['Text']
d1 = pd.read_table(train_dataL1, header=None, delimiter=None, names=names, engine='python-fwf')
d1['lang'] = 0

print("Pt-Br Data")
print("len: ", len(d1))
print(d1.head())

names = ['Text']
d2 = pd.read_csv(train_dataL2, header=None, delimiter=None, names=names, engine='python-fwf')
d2['lang'] = 1

print("Pt-Pt Data")
print("len: ", len(d2))
print(d2.head())


print("Len Pt-Br, Pt-Pt data: ", len(d1), len(d2))

# Concatinating the dataframes fro creating the training dataset
dt = [d1, d2]
df = pd.concat(dt)

# randomly shuffling the dataframe
df = df.sample(frac=1).reset_index(drop=True)
print("Shuffled Data: ")
print(df.head())


print("Len of complete training set: ", len(df))


print("Langauage Tag counts: ", df['lang'].value_counts())
# {pt-br pt-pt}


# Preprocessing of the text, took care of basic pre-processing task only.
def processSent(sent):
    sent = str(sent)
    # To lowercase
    sent = sent.lower()
    #remove @username
    sent = re.sub(r'@(\w+)','',sent)
    
    # Remove hashtags
    sent = re.sub(r'#(\w+)', '', sent)
    
    # Remove Punctuation and split 's, 't, 've with a space for filter    
    sent = re.sub(r"[-()\"«/;»:<>{}`+=~|.!&?',]", "", sent)
    
    # remove the numerical values
    sent = re.sub(r"(\s\d+)","",sent) 
    
    # Remove HTML special entities (e.g. &amp;)
    sent = re.sub(r'\&\w*;', '', sent)
    
    # Remove tickers
    sent = re.sub(r'\$\w*', '', sent)
    
    # Remove hyperlinks
    sent = re.sub(r'https?:\/\/.*\/\w*', '', sent)
   
    # Remove whitespace (including new line characters)
    sent = re.sub(r'\s\s+', ' ', sent)
   
    # Remove single space remaining at the front of the sent.
    sent = sent.lstrip(' ') 
    
    # Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
    sent = ''.join(c for c in sent if c <= '\uFFFF') 
    
    return sent

# Copya and process the sentences
data = df.copy()
data['Text'] = data['Text'].apply(processSent)
print(data.head())

# loading the testing data
names = ['Text']
dft = pd.read_table(test_data, header=None, delimiter=None, names=names)
print("Testing data details: ")
print("len", len(dft))
print("Test dataframe", dft.head())

# Pre-process the test data
dft['Text'] = dft['Text'].apply(processSent)

print("Preprocessed test data: ", dft.head())

# Creating our feature vector and classification model 
from sklearn.feature_extraction.text import TfidfVectorizer
train = list(data['Text'])
test = list(dft['Text'])

# Corpus so as to represent the entire entries n both train and test and their representation  
corpus = train+test
tfidf = TfidfVectorizer(max_features = 6000) 
tfidf.fit(corpus)
tfidf_features_train = tfidf.transform(train)
tfidf_features_test = tfidf.transform(test)

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support


# Converting the labels from strings to binary (if we have string 'lang' labels)
label = data['lang']
le = LabelEncoder()
le.fit(label)
label = le.transform(label)


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB

X_train = tfidf_features_train
Y_train = label

X_test = tfidf_features_test

print("Train data dimensions: ", (X_train.shape), (Y_train.shape))
print("Test data dimensions: ", (X_test.shape))

tfidf_model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

# tfidf_model = MultinomialNB()  
tfidf_model.fit(X_train, Y_train)
tfidf_prediction = tfidf_model.predict(X_test)

tfidf_prediction = np.nan_to_num(tfidf_prediction, copy=False).astype(np.int)
# {0: pt-br, 1: pt-pt}

# Saving our test prediction results
f = open('test_result.txt', 'w')

for i in range(len(tfidf_prediction)):
    if (tfidf_prediction[i]==0):
        f.write("pt-br"+"\n")
    elif (tfidf_prediction[i]==1):
        f.write("pt-pt"+"\n")

f.close()


# clean the sampled data
os.system("rm -rf "+ str(train_dataL1) + " " + str(train_dataL2))

print("Done training and testing, find the result in test_result.txt file")

# =====================================================================================
