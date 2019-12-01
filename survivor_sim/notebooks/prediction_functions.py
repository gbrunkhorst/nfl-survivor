# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 16:18:36 2019

@author: greg

Prediction Functions
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

# this is the prediction function for logistic regression only
def log_reg(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    lr = LogisticRegression().fit(X_train, y_train)
    
    percent_correct = sum(lr.predict(X_test)==y_test)/len(y_test)
    ll = log_loss(y_pred=lr.predict_proba(X_test), y_true=y_test)
    print('percent_correct = ',percent_correct*100)
    print('log_loss = ', ll)

# this is the prediction function using a few different algorithms
def bench_marks(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    models = {'logistic regression': LogisticRegression().fit(X_train, y_train), 
              'decision tree': DecisionTreeClassifier().fit(X_train, y_train),
              'random forest': RandomForestClassifier().fit(X_train, y_train),
              'extra trees': ExtraTreesClassifier().fit(X_train, y_train), 
              'linear support vector': LinearSVC().fit(X_train, y_train), 
             'support vector': SVC().fit(X_train, y_train)}
    results_percents = []
    results_ll = []
    
    for model_name, model in models.items():
        print(model_name,'---------')
        percent_correct = sum(model.predict(X_test)==y_test)/len(y_test)
        percent_correct = round(percent_correct*100,1)
        print('     percent_correct = ',percent_correct)
        try: 
            ll = log_loss(y_pred=model.predict_proba(X_test), y_true=y_test)
            ll = round(ll,2)
            print('     log_loss = ', ll)
        except:
            ll = None
            pass
        results_percents.append(percent_correct)
        results_ll.append(ll)
    return models, results_percents, results_ll

def rand_for(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=10, max_features='sqrt', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=4, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=1600, n_jobs=-1,
            oob_score=False, verbose=0,
            warm_start=False, random_state=42)
    clf.fit(X_train, y_train) 
    
    percent_correct = sum(clf.predict(X_test)==y_test)/len(y_test)
    ll = log_loss(y_pred = clf.predict_proba(X_test), y_true=y_test)
    print('    percent_correct = ',percent_correct*100)
    print('    log_loss = ', ll)
