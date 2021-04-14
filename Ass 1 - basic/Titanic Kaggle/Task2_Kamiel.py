import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split,cross_validate,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import AdaBoostClassifier
import Cleaner
from sklearn.pipeline import Pipeline
from scipy.cluster import hierarchy
from collections import defaultdict
from collections import defaultdict
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler

from scipy.stats import spearmanr
from scipy.cluster import hierarchy

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.stats
import Competition

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return round(m,3), round(m-h,3), round(m+h,3)
    
df1 = pd.read_csv("Ass 1 - basic/Titanic Kaggle/Data/train.csv")
df2 = pd.read_csv("Ass 1 - basic/Titanic Kaggle/Data/test.csv")

frames = [df1, df2]
df = pd.concat(frames)

# print(df)

print(df["Age"].isna().sum())
print((df["Age"].isna().sum())/len((df["Age"])))


# df = Cleaner.replace_titles(df)
Cleaner.get_deck(df)
Cleaner.Embarked_2(df)
Cleaner.family_size(df)
Cleaner.is_alone(df)
Cleaner.replace_titles(df)
Cleaner.title_num(df)
Cleaner.Class(df)
Cleaner.Binary_Sex(df)
Cleaner.Binary_cabin(df)
Cleaner.SexClass(df)
Cleaner.family_size(df)

Cleaner.fill_age(df)
Cleaner.age_class(df)
Cleaner.AgeClass(df)
df['Fare'] = df['Fare'].fillna(0)

non_num_features = ["Deck","Title","Age","Fare","Embarked","Family_Size_grouped"]
for feature in non_num_features:
    df[feature] = LabelEncoder().fit_transform(df[feature])


cat_features = ['Pclass', 'Sex', 'Deck', 'Embarked', 'Title', 'Family_Size_grouped']
encoded_features = []

for feature in cat_features:
    encoded_feat = OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()
    n = df[feature].nunique()
    cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
    encoded_df = pd.DataFrame(encoded_feat, columns=cols)
    encoded_df.index = df.index
    encoded_features.append(encoded_df)

df = pd.concat([df, *encoded_features[:6]], axis=1)

# df = Cleaner.replace_titles(df)
df_test = df.iloc[891:]
df =  df.iloc[:891]


# l = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Family_Size', 'Family_Size_grouped', 'Is_alone', 'Title', 'Title_num', 'Binary_Sex', 'Cabin_Binary', 'SexClass', 'Age_div', 'AgeClass', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_1', 'Sex_2', 'Deck_1', 'Deck_2', 'Deck_3', 'Deck_4', 'Embarked_1', 'Embarked_2', 'Embarked_3', 'Title_1', 'Title_2', 'Title_3', 'Title_4', 'Family_Size_grouped_1', 'Family_Size_grouped_2', 'Family_Size_grouped_3', 'Family_Size_grouped_4']
# # new_df = new_df.dropna()
# features = ["Fare", "Age","SibSp","Binary_Sex", "Parch", "Cabin_Binary","SexClass",'Age_div',"AgeClass","Title_num", "Family_Size"]
# x = new_df[["Binary_Sex","Fare","SexClass","AgeClass","Title_num", "Family_Size","SibSp","Pclass"]]
x = df[['Title_num', 'Sex_2', 'Fare', 'Age', 'Sex_1', 'Pclass_1','Pclass_2','Pclass_3', 'Pclass_3', 'Family_Size']]
y = df["Survived"]


X_train, X_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size=0.2, random_state=11)


pipeline = Pipeline([('scale', StandardScaler()),
    ('classifier', RandomForestClassifier(criterion="gini",  n_estimators=100, min_samples_leaf=4, max_depth=None, random_state=0))
])

Competition.make_submission(df, df_test, pipeline,"Kamiel")

pipeline1 = Pipeline([
    ("scalar",StandardScaler()),("classifier", SVC(kernel="rbf",gamma="scale",degree = 0.1, probability= True, decision_function_shape = 'ovr',random_state=0))
])

Competition.make_submission(df, df_test, pipeline1,"Dante")


# forest = RandomForestClassifier(criterion="gini",  n_estimators=75, min_samples_leaf=4, max_depth=None, random_state=0)
# forest.fit(x,y)
# importances = forest.feature_importances_
# std = np.std([tree.feature_importances_ for tree in forest.estimators_],
#              axis=0)
# indices = np.argsort(importances)[::-1]

# # Print the feature ranking
# print("Feature ranking:")
# s = []
# for f in range(x.shape[1]):
#     s.append(l[indices[f]])
#     if f == 10: 
#         print(s)
#         raise ValueError()


# # Plot the impurity-based feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(x.shape[1]), importances[indices],
#         color="r", yerr=std[indices], align="center")
# plt.xticks(range(x.shape[1]), indices)
# plt.xlim([-1, x.shape[1]])
# plt.show()
# clf = RandomForestClassifier()
# y_pred = clf.fit(X_train, y_train).predict(X_test)
# print(clf.score(X_test, y_test))
# print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

# print(cross_validate(pipeline, X_train, y_train, cv=10)['test_score'].mean())
# print(cross_validate(pipeline, X_test, y_test, cv=10)['test_score'].mean())
print("test, RF", mean_confidence_interval(cross_validate(pipeline, X_test, y_test, cv=10)['test_score']))
print("train, RF", mean_confidence_interval(cross_validate(pipeline, X_train, y_train, cv=10)['test_score']))

print("test, SVC", mean_confidence_interval(cross_validate(pipeline1, X_test, y_test, cv=10)['test_score']))
print("train, SVC", mean_confidence_interval(cross_validate(pipeline1, X_train, y_train, cv=10)['test_score']))

print(scipy.stats.ttest_ind((cross_validate(pipeline1, X_train, y_train, cv=10)['test_score']), cross_validate(pipeline, X_train, y_train, cv=10)['test_score']))
print(scipy.stats.ttest_ind((cross_validate(pipeline1, X_test, y_test, cv=10)['test_score']), cross_validate(pipeline, X_train, y_train, cv=10)['test_score']))
hyperparameters = {                     
                    'classifier__n_estimators': [25,50,75,100,500],
                    'classifier__max_depth': [None,2, 4],
                    'classifier__min_samples_leaf': [2, 4],
                    'classifier__criterion': ['gini', 'entropy'],

                }

# hyperparameters1 = {
#     'classifier__kernel': ["rbf","poly","sigmoid","linear"],
#     'classifier__gamma': ["scale","auto"],
#     'classifier__degree': [0.1,0.5,1,2,3],
#     'classifier__decision_function_shape': ["ovr","ovo"],
#     'classifier__probability' : [True, False]
# }


clf = GridSearchCV(pipeline, hyperparameters, cv = 10)
# Fit and tune model
clf.fit(X_train, y_train)


print(clf.best_params_)

# refitting on entire training data using best settings
# clf.refit

print(cross_validate(clf, X_test, y_test, cv=3)['test_score'].mean())


# raise ValueError


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
