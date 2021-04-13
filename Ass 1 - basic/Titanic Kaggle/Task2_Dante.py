import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import Cleaner

Titan_data = pd.read_csv("Ass 1 - basic/Titanic Kaggle/Data/train.csv")

""" 2A """
def some_plots(Titan_data):
    print(Titan_data.dtypes)

    sns.boxplot(y = Titan_data["Age"], x = Titan_data["Pclass"])
    plt.show()
    
    # Age distribution
    sns.histplot(Titan_data["Age"])
    plt.show()

    # Price ticket according to sex
    Titan_data_m = Titan_data.loc[Titan_data["Sex"] == "male"]
    Titan_data_f = Titan_data.loc[Titan_data["Sex"] == "female"]

    sns.distplot(Titan_data_m["Fare"], label="Male")
    sns.distplot(Titan_data_f["Fare"], label="Female")
    plt.legend()
    plt.xlim(0,max(Titan_data["Fare"]))
    plt.show()

    # Age vs Fare colored on Pclass
    sns.scatterplot(x=Titan_data["Age"],y=Titan_data["Fare"], hue = Titan_data["Pclass"])
    plt.show()

    sns.scatterplot(x=Titan_data["Age"],y=Titan_data["Fare"], hue = Titan_data["Survived"])
    plt.show()

    # Frequency tables
    Freq_table1 = pd.crosstab(index=Titan_data['SibSp'], columns=Titan_data['Parch'])
    print(Freq_table1,"\n")

    Freq_table2 = pd.crosstab(index=Titan_data['Survived'], columns=Titan_data['Pclass'])
    print(Freq_table2, "\n")
    print(Freq_table2/Freq_table2.sum(),"\n")

    Freq_table3 = pd.crosstab(index=Titan_data['Pclass'], columns=Titan_data['Embarked'])
    print(Freq_table3,"\n")

# some_plots(Titan_data)


""" 2B """
# Stratified sampling
Titan_split_X = Titan_data.drop(columns=["Survived"])
Titan_split_y = Titan_data["Survived"]



df = Titan_data

# Cleaner.Embarked(df)
Cleaner.Binary_Sex(df)
Cleaner.fill_age(df)
Cleaner.get_deck(df)
Cleaner.replace_titles(df)
Cleaner.title_num(df)
Cleaner.Binary_cabin(df)
Cleaner.family_size(df)
Cleaner.is_alone(df)
Cleaner.age_class(df)
df = df.drop(columns=["Name"])
df = df.drop(columns=["Sex"])
df = df.drop(columns=["Ticket"])
df = df.drop(columns=["Cabin"])

# print(df.columns)

# print(df["Title"].unique())

print(df.head())

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

x = df[["Pclass","PassengerId","Title_num","Binary_Sex","Family_Size","Age_div","Fare","Is_alone"]]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size = 0.6, test_size=0.2, random_state=5)

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
print(cross_validate(clf, x, y, cv=10)['test_score'].mean())

N_splits = 5
sss = StratifiedShuffleSplit(n_splits=N_splits, test_size=0.5, random_state=0)
score = 0
for train_index, test_index in sss.split(x, y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    clf.fit(X_train,y_train)
    score += clf.score(X_test,y_test)
print(score/N_splits)

# from scipy.stats import spearmanr
# from scipy.cluster import hierarchy
# from sklearn.inspection import permutation_importance
# from collections import defaultdict

# clf.fit(X_train, y_train)
# print("Accuracy on test data: {:.2f}".format(clf.score(X_test, y_test)))

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
# corr = spearmanr(x).correlation
# corr_linkage = hierarchy.ward(corr)

# cluster_ids = hierarchy.fcluster(corr_linkage, 1, criterion='distance')
# cluster_id_to_feature_ids = defaultdict(list)
# for idx, cluster_id in enumerate(cluster_ids):
#     cluster_id_to_feature_ids[cluster_id].append(idx)
# selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]

# X_train_sel = X_train.iloc[:, selected_features]
# X_test_sel = X_test.iloc[:, selected_features]

# clf_sel = clf
# clf_sel.fit(X_train_sel, y_train)
# print("Accuracy on test data with features removed: {:.2f}".format(
#       clf_sel.score(X_test_sel, y_test)))