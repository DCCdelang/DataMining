# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 09:41:21 2021

@author: Gebruiker
"""
import pandas as pd 
import numpy as np 
import math
import seaborn as sns
import Data_cleaner
import matplotlib.pyplot as plt
from scipy import stats
import collections


ODI_data = pd.read_csv("Data/ODI-2021.csv")

# print(ODI_data.head())

print("Amount of colomns = ", ODI_data.shape[1])
print("Amount of answers = ", ODI_data.shape[0])


def chocolate_gender(df):    
    """
    Looks at how people from different gender look towards chocolate
    """
    sns.catplot(x="Gender", hue="Chocolate", kind="count", data=df)
    plt.show()


def stress_esteam(df):
    """
    Looks at the relation between stress and the amount of money one would think
    he/she would earn when doing DM
    """


    print(stats.spearmanr(df["Stress_c"], df["Self esteem_c"]))
    sns.scatterplot(data=df, x="Stress_c", y="Self esteem_c")
    plt.show()

def stress_check(df, course):
    """
    Checks the difference in stress between a group who did do a course
    and a group who did not do that course
    """
    df = Data_cleaner.make_numeric(df,"Stress")
    df = Data_cleaner.remove_nan(df)
    df = Data_cleaner.remove_numeric_values(df,"Stress",100, 0)

    sns.catplot(x=course, y="Stress", data=df)
    plt.show()

    
    yes = [item for item, count in collections.Counter(df[course]).items() if count > 1][0]
    no = [item for item, count in collections.Counter(df[course]).items() if count > 1][1]

    x = df['Stress'][df[course] == yes] 
    y = df['Stress'][df[course] == no]



    print(f"x ={x.mean()}, y = {y.mean()}")
    print(stats.mannwhitneyu(x, y))


if __name__ == "__main__":

    # Import data
    df = ODI_data
  
    # Make new collumn names
    new_cols = ["Time", "Programme", "ML", "IR", "Stat", "DB","Gender","Chocolate","Birthday","Neighbours", "Stand up", "Stress", "Self esteem", "RN", "Bedtime","GD1", "GD2"]
    df = Data_cleaner.rename_collumns(df,new_cols)
    # df = Data_cleaner.stress_cleaner(df)
    # df = Data_cleaner.se_cleaner(df)
    # df = df.dropna()
    # Data_cleaner.RN_cleaner(df)
    # Activate functions
    # stress_check(df, "IR")
    # stress_esteam(df)
    # chocolate_gender(df)
    Data_cleaner.programme_cleaner(df)
    
    Data_cleaner.birth_date_cleaner(df)
    print(df["Programme_c"].head(100))
    print(df["Programme"].head(100))
    sns.catplot(x="Programme_c", kind="count", data=df)
    plt.show()

 