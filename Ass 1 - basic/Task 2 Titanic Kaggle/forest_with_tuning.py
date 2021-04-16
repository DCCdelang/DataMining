# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 09:59:08 2021

@author: Gebruiker
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
from sklearn import tree

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

def run_forest(df, y, test_size):
    pipeline = Pipeline([

        ('regressor', RandomForestRegressor(random_state = 3,n_estimators = 1000 ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=test_size) #random_state=seed
    
    pipeline.fit(X_train, y_train)
    predicts = pipeline.predict(X_test)
    
    mae = mean_absolute_error(predicts, y_test)
    #mape = np.mean(np.abs((y_test - predicts) / np.abs(y_test)))
    #print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
    #print('Accuracy:', round(100*(1 - mape), 2))
    
    print('mae', mae)
    print('r2',metrics.r2_score(y_test, predicts))
    cross_val = (cross_val_score(pipeline['regressor'], df, y, cv=10))
    
    print(cross_val.sum())
    
    

    #tuning the model  
    hyperparameters = { 'features__text__tfidf__max_df': [0.9, 0.95],
                    'features__text__tfidf__ngram_range': [(1,1), (1,2)],
                    'classifier__learning_rate': [0.1, 0.2],
                    'classifier__n_estimators': [50,100, 1000],
                    'classifier__max_depth': [2, 4],
                    'classifier__min_samples_leaf': [2, 4],
                    'classifier__min_samples_leaf': [2, 4]
                  }
    clf = GridSearchCV(pipeline, hyperparameters, cv = 3)
    # Fit and tune model
    clf.fit(X_train, y_train)
    
    print(clf.best_params_)
    
    #refitting on entire training data using best settings
    clf.refit

    predicts = clf.predict(X_test)
    
    mae = mean_absolute_error(predicts, y_test)
    #mape = np.mean(np.abs((y_test - predicts) / np.abs(y_test)))
    #print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
    #print('Accuracy:', round(100*(1 - mape), 2))
    #accuracy = (100*(1-mape))
    print('mae', mae)
    print('r2',metrics.r2_score(y_test, predicts))


    
    
    