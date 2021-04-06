import pandas as pd 
import numpy as np 
import math
import seaborn as sns
from Data_cleaner import Data_cleaner
import matplotlib.pyplot as plt
from scipy import stats
import collections


ODI_data = pd.read_csv("Ass 1 - basic/Data/ODI-2021.csv")

print(ODI_data.head())

print("Amount of colomns = ", ODI_data.shape[1])
print("Amount of answers = ", ODI_data.shape[0])





## [1, 2, 5]

# sns.displot(ODI_data,x="What is your stress level (0-100)?", col = "What is your gender?", row = "Did you stand up?",binwidth=3, height=3, facet_kws=dict(margin_titles=True))
# plt.show()



df = ODI_data

new_cols = ["Time", "Programme", "ML", "IR", "Stat", "DB","Gender","Chocolate","Birthday","Neighbours", "Stand up", "Stress", "Self esteem", "RN", "Bedtime","GD1", "GD2"]


df = Data_cleaner(df).rename_collumns(new_cols)

def chocolate_gender(df):    
    """
    Looks at how people from different gender look towards chocolate
    """
    sns.catplot(x="Gender", hue="Chocolate", kind="count", data=df)
    plt.show()
# print(stats.shapiro(df["Self esteem"]))

def stress_esteam(df):
    """
    Looks at the relation between stress and the amount of money one would think
    he/she would earn when doing DM
    """
    l = ["Stress", "Self esteem"]
    for i in l:
        df = Data_cleaner(df).make_numeric(i)
        df = Data_cleaner(df).remove_nan()
        df = Data_cleaner(df).remove_numeric_values(i,100, 0)


    print(stats.spearmanr(df["Stress"], df["Self esteem"]))
    sns.scatterplot(data=df, x="Stress", y="Self esteem")
    plt.show()

def stress_check(df, course):
    """
    Checks the difference in stress between a group who did do a course
    and a group who did not do that course
    """
    df = Data_cleaner(df).make_numeric("Stress")
    df = Data_cleaner(df).remove_nan()
    df = Data_cleaner(df).remove_numeric_values("Stress",100, 0)

    sns.catplot(x=course, y="Stress", data=df)
    plt.show()

    
    yes = [item for item, count in collections.Counter(df[course]).items() if count > 1][0]
    no = [item for item, count in collections.Counter(df[course]).items() if count > 1][1]

    x = df['Stress'][df[course] == yes] 
    y = df['Stress'][df[course] == no]



    print(f"x ={x.mean()}, y = {y.mean()}")
    print(stats.mannwhitneyu(x, y))

stress_check(df, "IR")
"""
TODO
- Set new colomns labels
- Filter data and make it neat
- Show outliers
- Make neat plots for the categorical colomns
- 
"""
