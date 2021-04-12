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

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.33, random_state=0)
# print(sss.get_n_splits(Titan_split_X, Titan_split_y))

# for train_index, test_index in sss.split(Titan_split_X, Titan_split_y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    # print(len(train_index),len(test_index))
    # print(Titan_split_X[train_index])
    # X_train, X_test = Titan_split_X[train_index], Titan_split_X[test_index]
    # y_train, y_test = Titan_split_y[train_index], Titan_split_y[test_index]

df = Titan_data

Cleaner.Embarked(df)
Cleaner.Binary_Sex(df)
Cleaner.fill_age(df)
Cleaner.get_deck(df)
Cleaner.replace_titles(df)
Cleaner.title_num(df)
Cleaner.family_size(df)
Cleaner.is_alone(df)
Cleaner.age_class(df)
df = df.drop(columns=["Name"])
df = df.drop(columns=["Sex"])
df = df.drop(columns=["Ticket"])
df = df.drop(columns=["Cabin"])

print(df["Title"].unique())

print(df.head())
