import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import AdaBoostClassifier
import Cleaner
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from scipy.cluster import hierarchy
from collections import defaultdict
from collections import defaultdict
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from scipy.cluster import hierarchy

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

df = pd.read_csv("Ass 1 - basic/Titanic Kaggle/Data/train.csv")
df = Cleaner.Embarked(df)
df = Cleaner.Class(df)


# df = Cleaner.replace_titles(df)
df = Cleaner.Class(df)
df = Cleaner.Binary_Sex(df)
df = Cleaner.Binary_cabin(df)
df = Cleaner.SexClass(df)
df = Cleaner.family_size(df)
df = Cleaner.fill_age(df)
df = Cleaner.age_class(df)
df = Cleaner.AgeClass(df)
df = Cleaner.replace_titles(df)
df = Cleaner.title_num(df)

Cleaner.family_size(df)
Cleaner.is_alone(df)
# df = Cleaner.replace_titles(df)


print(df.columns)

# new_df = new_df.dropna()
features = ["Fare", "Age","SibSp", "Survived","Binary_Sex", "Parch","C1","C2","C3", "Cabin_Binary","SexClass",'C', 'Q', 'S', '0','Age_div',"AgeClass","Title_num", "Family_Size"]
# x = new_df[["Binary_Sex","Fare","SexClass","AgeClass","Title_num", "Family_Size","SibSp","Pclass"]]
x = df[["Pclass","PassengerId","Title_num","Binary_Sex","Family_Size","Age_div","Fare","Is_alone"]]
y = df["Survived"]


X_train, X_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size=0.2, random_state=11)
pipeline = Pipeline([('scale', StandardScaler()),
    ('classifier', RandomForestClassifier(criterion="entropy",  n_estimators=100, min_samples_leaf=2, max_depth=None, random_state=0))
])



# clf = RandomForestClassifier()
# y_pred = clf.fit(X_train, y_train).predict(X_test)
# print(clf.score(X_test, y_test))
# print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

print(cross_validate(pipeline, X_test, y_test, cv=10)['test_score'].mean())

hyperparameters = {                     
                    'classifier__n_estimators': [25,50,75,100,500],
                    'classifier__max_depth': [None,2, 4],
                    'classifier__min_samples_leaf': [2, 4],
                    'classifier__criterion': ['gini', 'entropy'],

                }

clf = GridSearchCV(pipeline, hyperparameters, cv = 10)
# Fit and tune model
clf.fit(X_train, y_train)


print(clf.best_params_)

# refitting on entire training data using best settings
clf.refit

print(cross_validate(clf, X_test, y_test, cv=3)['test_score'].mean())





clf = ExtraTreesClassifier(n_estimators=50, max_depth=None, min_samples_split=2, random_state=0)
y_pred = clf.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

print(cross_validate(clf, x, y, cv=10)['test_score'].mean())

clf = AdaBoostClassifier(n_estimators=1000)
y_pred = clf.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

print(cross_validate(clf, x, y, cv=10)['test_score'].mean())


# clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print(clf.score(X_test, y_test))
# print(cross_validate(clf, x, y, cv=10)['test_score'].mean())

clf = make_pipeline(StandardScaler(),
                    LinearSVC(random_state=0))
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
print(cross_validate(clf, x, y, cv=10)['test_score'].mean())

clf = KNeighborsClassifier(n_neighbors=5)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(cross_validate(clf, x, y, cv=10)['test_score'].mean())

clf  = GaussianNB()
y_pred = clf .fit(X_train, y_train).predict(X_test)
# print(y_pred)

# print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
print(cross_validate(clf , x, y, cv=10)['test_score'].mean())
