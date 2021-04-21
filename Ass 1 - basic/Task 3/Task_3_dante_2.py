import pandas as pd
import numpy as np 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import scipy

df = pd.read_csv("Task 3/sms_vec.csv")

x = df.drop(["Label"],axis=1)

y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size = 0.6, random_state=0)

clf = SVC(degree=1,gamma="scale",kernel="sigmoid")

clf = SVC()

clf.fit(X_train, y_train)

# print(cross_validate(clf, X_test, y_test, cv=10)['test_score'].mean())


feature_scaler = StandardScaler()
pipeline1 = Pipeline([
    ("scalar",feature_scaler),("classifier", clf)
])

print(cross_validate(pipeline1, X_test, y_test, cv=10)['test_score'].mean())

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return round(m,3), round(m-h,3), round(m+h,3)

print(mean_confidence_interval(cross_validate(pipeline1, X_test, y_test, cv=10)['test_score']))

raise ValueError

hyperparam1 = {
    'classifier__kernel': ["rbf","poly","sigmoid"],
    'classifier__gamma': ["scale","auto"],
    'classifier__degree': [0.1,0.5,1,2,3]
}

clf = GridSearchCV(pipeline1, hyperparam1, cv = 5, verbose=3)

# Fit and tune model
clf.fit(X_train, y_train)

print(clf.best_params_)

print(cross_validate(clf, X_test, y_test, cv=10)['test_score'].mean())