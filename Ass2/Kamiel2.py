import pandas as pd
import random
import numpy as np
import seaborn as sns
import operator
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import ndcg_score, make_scorer, SCORERS, accuracy_score, classification_report, precision_score,label_ranking_average_precision_score, make_scorer
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier, StackingRegressor, AdaBoostRegressor, GradientBoostingRegressor,GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.model_selection import GridSearchCV 
# from LambdaRankNN import LambdaRankNN
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.impute import KNNImputer

  
train = pd.read_csv('Data/validation_test.csv')
imputer = KNNImputer(n_neighbors=2)
print('start imputing')
train.loc[:, train.columns == 'prop_location_score2'] = imputer.fit_transform(train.loc[:, train.columns == 'prop_location_score2'])
train.to_csv('test_validation_KNN.csv')

# test = pd.read_csv('Data/test_set_VU_DM.csv')
# train = pd.read_csv('Data/training_set_VU_DM.csv')

# train1 = train
# train1 = train1.drop(['booking_bool', 'click_bool', 'position', 'gross_bookings_usd'],axis=1)

total = pd.read_csv('Data/concated.csv')

print(total.describe())



