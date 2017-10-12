#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 13:01:12 2017

@author: rahulrksmau
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

train= pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train = train.drop(['Description','User_ID'], axis=1)
test = test.drop(['Description','User_ID'] , axis=1)
features = ['Browser_Used','Device_Used','Is_Response']

for f in features:
    encode = LabelEncoder()
    train[f] = encode.fit_transform(train[f])

    
for f in features[:-1]:
    encode = LabelEncoder()
    test[f] = encode.fit_transform(test[f])


X = train.drop('Is_Response',axis=1) #feature  (38932, 3)
X = pd.concat([X,test]).reset_index(drop=True)
y = train['Is_Response'] # label  (38932,)


import pickle

with open('loader.pk1', 'rb') as f:
    clf = pickle.load(f)

c_df = pd.DataFrame(clf.todense()) # shape 68336*500
c_df.columns = ['column'+str(x) for x in c_df.columns]

for col in X.columns:
    c_df[col] = X[col]
    
with open('clss.pk1','rb') as clf:
    clss = pickle.load(clf)
    
predict = np.array(clss.predict(c_df[len(train):]))
predict = np.where(predict, 'happy','not happy')
submission = pd.read_csv('sample_submission.csv')
Ids = submission['User_ID']
output = pd.DataFrame({'User_ID':Ids, 'Is_Response': predict})
output.to_csv('sub.csv',index=False)


