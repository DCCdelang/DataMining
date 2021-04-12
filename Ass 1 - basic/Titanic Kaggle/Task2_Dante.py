import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

Titan_data = pd.read_csv("Ass 1 - basic/Titanic Kaggle/Data/train.csv")



""" 2A """
def some_plots(Titan_data):
    print(Titan_data.dtypes)

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

""" Transformation on original dataset """

df = Titan_data

def get_deck(df):
    floor_list = []
    for cabin in df["Cabin"]:
        if str(cabin)[0] != "n":
            floor_list.append(str(cabin)[0])
        else:
            floor_list.append(np.nan)
    df["deck"] = floor_list

get_deck(df)

def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if big_string.find(substring) != -1:
            return substring
    return np.nan

def replace_titles(df):
    title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                    'Don', 'Jonkheer']

    df['Title']=df['Name'].map(lambda x: substrings_in_string(x, title_list))

    titles = []
    for title in df['Title']:
        if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            return 'Mr'
        elif title in ['Countess', 'Mme']:
            return 'Mrs'
        elif title in ['Mlle', 'Ms']:
            return 'Miss'
        elif title =='Dr':
            if df['Sex']=='Male':
                return 'Mr'
            else:
                return 'Mrs'
        else:
            titles.append(title)

    df['Title'] = titles
    return df

replace_titles(Titan_data)
print(Titan_data)

""" 2B """
# Stratified sampling
Titan_split_X = Titan_data.drop(columns=["Survived"])
Titan_split_y = Titan_data["Survived"]

<<<<<<< HEAD
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
sss.get_n_splits(Titan_split_X, Titan_split_y)
=======
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.33, random_state=0)
print(sss.get_n_splits(Titan_split_X, Titan_split_y))

# for train_index, test_index in sss.split(Titan_split_X, Titan_split_y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    # print(len(train_index),len(test_index))
    # print(Titan_split_X[train_index])
    # X_train, X_test = Titan_split_X[train_index], Titan_split_X[test_index]
    # y_train, y_test = Titan_split_y[train_index], Titan_split_y[test_index]
>>>>>>> 783677baf68618a198c55fc1cb6942aafc8cf1ab
