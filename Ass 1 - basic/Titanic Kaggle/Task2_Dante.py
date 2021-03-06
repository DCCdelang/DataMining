import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

Titan_data = pd.read_csv("Data/train.csv")

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

some_plots(Titan_data)

""" Transformation on original dataset """

df = Titan_data

def get_deck(df):
    floor_list = []
    for cabin in df["Cabin"]:
        if cabin:
            floor_list.append(str(cabin)[0])
        else:
            floor_list.append(np.nan)
    df["deck"] = floor_list

get_deck(df)

""" 2B """
# Stratified sampling
Titan_data_validation = pd.read_csv("Data/test.csv")

Titan_split_X = Titan_data.drop(columns=["Survived"])
Titan_split_y = Titan_data["Survived"]

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
sss.get_n_splits(Titan_split_X, Titan_split_y)
