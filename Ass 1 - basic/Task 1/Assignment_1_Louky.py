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

import Categorisations
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None  # default='warn'


# ODI_data = pd.read_csv("Data/ODI-2021.csv")
# ODI_data = pd.read_csv("Data/ODI-2021.csv")

# print(ODI_data.head())

# print("Amount of colomns = ", ODI_data.shape[1])
# print("Amount of answers = ", ODI_data.shape[0])


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
    
    
def pie_charts(df):
    # df['Programme_c'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    # plt.show()
    # df['ML'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    # plt.show()
    # df['IR'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    # plt.show()
    # df['Stat'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    # plt.show()
    # df['DB'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    # plt.show()
    # df['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    # plt.show()
    # df['Chocolate'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    # plt.show()
    # df['Stand up'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    # plt.show()
    
    #histograms
    df['Neighbors_c'].value_counts().plot(kind='hist')
    plt.show()
    df['Bedtime_Hour_c'].plot(kind='hist', bins = 30)
    plt.show()
    

    
    df = df[(df['Age']>10) & (df['Age']<40)]
    # df['Age'].plot(kind='hist', bins = 40)
    sns.histplot(data=df, x="Age")
    plt.xlabel("Age", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.legend(fontsize=14)
    print(stats.shapiro(df["Age"]))
    plt.tight_layout()
    plt.savefig("Age.pdf")
    plt.show()
    
def RN_cleaner2(df):
    df["RN_c"] = df["RN"]

    for i in df["RN"]:
        if i.isdigit():
            if float(i)>10:
                df["RN_c"] = df["RN_c"].replace(i,10)
            elif float(i)<0:
                df["RN_c"] = df["RN_c"].replace(i,0)
        else:
            df["RN_c"] = df["RN_c"].replace(i,0)
    df["RN_c"] = pd.to_numeric(df["RN_c"], errors='coerce', downcast='integer')

    #for i in df["RN_c"]:
        #print(i)
    return df

def binarize(df):
    df = df.replace({'Stat' : { 'mu' : 2, 'sigma' : 0, 'unknown' : 1}})
    df["Stat"] = pd.to_numeric(df["Stat"])
    #df = df.replace({'ML' : { 'yes' : 1, 'no' : 0}})
    df = df.replace({'DB' : { 'ja' : 2, 'nee' : 0, 'unknown' : 1}})
    df["DB"] = pd.to_numeric(df["DB"])
    df = df.replace({'ML' : { 'yes' : 2, 'no' : 0, 'unknown' : 1}})
    df["ML"] = pd.to_numeric(df["ML"])
    df = df.replace({'Stand up' : { 'yes' : 2, 'no' : 0, 'unknown' : 1}})
    df["Stand up"] = pd.to_numeric(df["Stand up"])
    df = df.replace({'Gender' : { 'female' : 2, 'male' : 0, 'unknown' : 1}})
    df["Gender"] = pd.to_numeric(df["Gender"])
    
    return df


def main():
        # Import data
    df = ODI_data
    

    # Make new collumn names
    new_cols = ["Time", "Programme", "ML", "IR", "Stat", "DB","Gender","Chocolate","Birthday","Neighbours", "Stand up", "Stress", "Self esteem", "RN", "Bedtime","GD1", "GD2"]
    
    
    df = Data_cleaner.rename_collumns(df,new_cols)
    df = Data_cleaner.stress_cleaner(df)
    df = Data_cleaner.se_cleaner(df)
    df = df.dropna()
    

    df = RN_cleaner2(df)
    # Activate functions
    # stress_check(df, "IR")
    # stress_esteam(df)
    # chocolate_gender(df)
    
    Data_cleaner.programme_cleaner(df)
    Data_cleaner.neighbors_cleaner(df)
    Data_cleaner.birth_date_cleaner(df)
    df = Data_cleaner.bedtime_parser(df)
    
    #time of filling out questionaire 
    df = Data_cleaner.time(df)
    
    print(df["Programme_c"].head(100))
    print(df["Programme"].head(100))
    #sns.catplot(x="Programme_c", kind="count", data=df)
    
    df = Data_cleaner.calc_age(df)
    df =  Data_cleaner.binarize(df)
    pie_charts(df)
    
    #df = binarize(df)
    
    y = df['Stress_c']
    
    new_df_binary = df[['ML', 'Self esteem_c', 'RN_c', 'Neighbors_c', 'Bedtime_Hour_c', 'Age', 'Time_c']]
    
    #one hot encoding
    
    """
    df_encode = df[['Chocolate', 'Programme', 'Gender', 'IR', 'Stand up', 'Stat', 'DB']]
    df_encoded = pd.get_dummies(df_encode.astype(str))
    
    new_df = pd.concat([new_df_binary, df_encoded], axis=1)
    """
    
    df_selection = df[['Chocolate', 'Programme_c', 'Gender']]
    df_encoded = pd.get_dummies(df_selection.astype(str))
    
    
    
    #perform forest with chocolate, programme and gender
    print("Small set")
    print("")
    Categorisations.svm_model(df_encoded, y, 0.2, fold = 5)

    Categorisations.svm_model(df_encoded, y, 0.2, fold = 10)

    
    print("Larger set")
    print()
    new_df = pd.concat([df[['Self esteem_c', 'RN_c', 'Neighbors_c','DB']], df_encoded],axis = 1)
    
    #random forest
    Categorisations.svm_model(new_df, y, 0.2, fold = 5)

    Categorisations.svm_model(new_df, y, 0.2, fold = 10)
    
    
    
    
    
    
    
if __name__ == "__main__":
    main()

