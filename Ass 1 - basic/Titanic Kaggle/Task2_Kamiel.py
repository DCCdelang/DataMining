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

def cleaner(df):
    df["Binary_Sex"] = df["Sex"]
    print(df["Binary_Sex"])
    df["Binary_Sex"] = df["Binary_Sex"].replace(["female", "male"], [1, 0])
    enc = OneHotEncoder(handle_unknown='ignore')# passing bridge-types-cat column (label encoded values of bridge_types)

    enc_df = pd.DataFrame(enc.fit_transform(df[["Pclass"]]).toarray())# merge with main df bridge_df on key values
    df = df.join(enc_df)
    for i in range(3):
        df = df.rename(columns={i:f"C{i+1}"})

    

    df["Cabin_Binary"] = df["Cabin"]
    for i in df["Cabin_Binary"]:

        if type(i) == str:
            df["Cabin_Binary"] = df["Cabin_Binary"].replace(i,1)
        else:
            df["Cabin_Binary"] = df["Cabin_Binary"].replace(i,0)
    return df 


df = pd.read_csv("Ass 1 - basic/Titanic Kaggle/Data/train.csv")
df = cleaner(df)
print(df.columns)
new_df = df[["Fare", "Age","SibSp", "Survived","Binary_Sex", "Parch","C1","C2","C3", "Cabin_Binary"]]
new_df = new_df.dropna()
x = new_df[["Fare","Age","Binary_Sex",  "Parch","C1","C2","C3", "Cabin_Binary"]]
y = new_df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size=0.2, random_state=11)


clf  = GaussianNB()
y_pred = clf .fit(X_train, y_train).predict(X_test)
# print(y_pred)

# print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
print(cross_validate(clf , x, y, cv=10)['test_score'].mean())

n_estimators = [50, 100]

for n in n_estimators:
    clf = RandomForestClassifier(random_state=10, n_estimators=n, criterion="entropy")
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    # print(clf.score(X_test, y_test))
    # print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

    print(cross_validate(clf, x, y, cv=10)['test_score'].mean())

clf = ExtraTreesClassifier(n_estimators=50, max_depth=None, min_samples_split=2, random_state=0)
y_pred = clf.fit(X_train, y_train).predict(X_test)
print(cross_validate(clf, x, y, cv=10)['test_score'].mean())

clf = AdaBoostClassifier(n_estimators=50)
y_pred = clf.fit(X_train, y_train).predict(X_test)
print(cross_validate(clf, x, y, cv=10)['test_score'].mean())


# clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print(clf.score(X_test, y_test))
# print(cross_validate(clf, x, y, cv=10)['test_score'].mean())

# clf = make_pipeline(StandardScaler(),
#                     LinearSVC(random_state=0))
# clf = clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print(cross_validate(clf, x, y, cv=10)['test_score'].mean())

# clf = KNeighborsClassifier(n_neighbors=5)
# clf = clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print(cross_validate(clf, x, y, cv=10)['test_score'].mean())

