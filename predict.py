#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 17:28:51 2017
@author: rahulrksmau
"""
import pandas as pd
from notebook1 import c_df,y
from sklearn.model_selection import cross_val_score, train_test_split

train_len = len(pd.read_csv('train.csv'))
test_len = len(pd.read_csv('test.csv'))
X_train, X_test, y_train, y_test=train_test_split(
                    c_df[:train_len],y, test_size=0.33, random_state=42)

'''        
#predict
import pickle
with open('KNN.pk1','wb') as KN:
    pickle.dump(bst_model, KN)

for j in ['linear', 'poly', 'rbf', 'sigmoid']:
    clf = svm.SVR(kernel=j)
    clf.fit(X_train, y_train)
    predict_svm = clf.score(X_test, y_test)
    print (j,predict_svm)


from sklearn import svm

clf = svm.SVR() # 0.77 accuracy
clf.fit(X_train, y_train)
predict_svm = clf.score(X_test, y_test)
print "prediction on basis of SVM Classifier ",
print predict_svm
for j in ['linear', 'poly', 'rbf', 'sigmoid']:
    clf = svm.SVR(kernel=j)
    clf.fit(X_train, y_train)
    predict_svm = clf.score(X_test, y_test)
    print (j,predict_svm)

import pickle
with open('load_clss.pk1','wb') as KN:
    pickle.dump(clf, KN)
'''

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
bst_score= 0
bst_model= KNeighborsClassifier()
for i in range(1,5):
    clfr = KNeighborsClassifier(n_neighbors=i)
    clfr.fit(X_train, y_train)
    predictions = clfr.predict(X_test)
    predict = accuracy_score(y_test, predictions)
    if  predict > bst_score: 
        bst_score = predict
        bst_model = clfr
        
print 'best score is  %s'%str(bst_score)
print ('best model is ',bst_model)       
        