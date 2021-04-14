import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import Cleaner
import Competition
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

Titan_data = pd.read_csv("Ass 1 - basic/Titanic Kaggle/Data/train.csv")
Titan_data_test = pd.read_csv("Ass 1 - basic/Titanic Kaggle/Data/test.csv")

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
def total_clean(df):
    Cleaner.Embarked_2(df)
    Cleaner.Binary_Sex(df)
    Cleaner.fill_age(df)
    Cleaner.get_deck(df)
    Cleaner.replace_titles(df)
    # Cleaner.title_num(df)
    Cleaner.Binary_cabin(df)
    Cleaner.family_size(df)
    Cleaner.is_alone(df)
    Cleaner.age_class(df)
    df = df.drop(columns=["Name"])
    df = df.drop(columns=["Sex"])
    df = df.drop(columns=["Ticket"])
    df = df.drop(columns=["Cabin"])
    df["Fare"] = df["Fare"].fillna(0)
    return df

df = total_clean(Titan_data)
df_test = total_clean(Titan_data_test)

non_num_features = ["Deck","Title","Age","Fare","Embarked"]
for feature in non_num_features:
    df[feature] = LabelEncoder().fit_transform(df[feature])

# print(df.columns)
# print(df.head())

# raise ValueError

x = df[["Pclass","Title","Binary_Sex","Family_Size","Age","Fare","Is_alone", "Deck", "Embarked"]]

y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size = 0.6, random_state=0)

# Train on everything
# X_train = x
# y_train = y
# X_test = df_test[["Pclass","PassengerId","Title_num","Binary_Sex","Family_Size","Age_div","Fare","Is_alone"]]

# Choose from StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
feature_scaler = StandardScaler()

from sklearn.metrics import plot_confusion_matrix

clf = SVC(kernel="linear",gamma="scale",degree = 0.1, probability= True, decision_function_shape = 'ovr',random_state=0)

clf.fit(X_train, y_train)

plot_confusion_matrix(clf, X_test, y_test)  
plt.show()  

# from sklearn.metrics import roc_curve
# y = np.array([1, 1, 2, 2])
# scores = np.array([0.1, 0.4, 0.35, 0.8])
# fpr, tpr, thresholds = roc_curve(y, scores, pos_label=2)

# plt.plot(fpr,tpr)
# plt.show()

# raise ValueError("You done!")

pipeline1 = Pipeline([
    ("scalar",feature_scaler),("classifier", SVC(kernel="linear",gamma="scale",degree = 0.1, probability= True, decision_function_shape = 'ovr',random_state=0))
])

pipeline2 = Pipeline([
    ("scalar",feature_scaler),("classifier", LinearSVC(random_state=0,max_iter=100000))
])

# only for test setting
print(cross_validate(pipeline1, X_test, y_test, cv=10)['test_score'].mean())

hyperparam1 = {
    'classifier__kernel': ["rbf","poly","sigmoid","linear"],
    'classifier__gamma': ["scale","auto"],
    'classifier__degree': [0.1,0.5,1,2,3],
    'classifier__decision_function_shape': ["ovr","ovo"],
    'classifier__probability' : [True, False]
}

hyperparam2 = {
    # 'classifier__penalty': ["l1","l2"],
    # 'classifier__loss': ["hinge","squared_hinge"],
    'classifier__dual' : [True,False],
    'classifier__C': [0.01,0.1,0.5,1],
    'classifier__multi_class': ["ovr","crammer_singer"]
}

cv_test= KFold(n_splits=5)
clf = GridSearchCV(pipeline1, hyperparam1, cv = cv_test)

# Fit and tune model
clf.fit(X_train, y_train)

print(clf.best_params_)

# only for train setting
print(cross_validate(clf, X_train, y_train, cv=10)['test_score'].mean())

# only for test setting
print(cross_validate(clf, X_test, y_test, cv=10)['test_score'].mean())

# Competition.make_submission(clf)

# print(clf.predict(X_test))

raise ValueError("You done!")

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