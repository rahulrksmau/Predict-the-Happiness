#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 16:35:52 2017

@author: rahulrksmau
"""

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer

data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
all_data = pd.concat([data_train, data_test]).reset_index(drop=True)
train_len = len(data_train)
test_len = len(data_test)

stops = set(stopwords.words("english"))

def cleanData(text, lowercase = False, remove_stops = False, stemming = False):
    txt = str(text)
    txt = re.sub(r'[^A-Za-z0-9\s]',r'',txt)
    txt = re.sub(r'\n',r' ',txt)
    
    if lowercase:
        txt = " ".join([w.lower() for w in txt.split()])
        
    if remove_stops:
        txt = " ".join([w for w in txt.split() if w not in stops])
    
    if stemming:
        st = PorterStemmer()
        txt = " ".join([st.stem(w) for w in txt.split()])

    return txt

all_data['Description'] = all_data['Description'].map(lambda x: cleanData(x,lowercase=True,
                              remove_stops=True, stemming=True)) 
#clf = CountVectorizer.fit_transform()


count = CountVectorizer(analyzer='word',   ngram_range= (1,1) , min_df=200, max_features=500)
clf = count.fit_transform(all_data['Description'])

import pickle
with open('loader.pk1', 'wb') as l:
    pickle.dump(clf,l)

