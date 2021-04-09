import pandas as pd 
import numpy as np 
import math
import seaborn as sns
import Data_cleaner
import Categorisations
import matplotlib.pyplot as plt
from scipy import stats
import collections


ODI_data = pd.read_csv("Ass 1 - basic/Data/ODI-2021.csv")

# print(ODI_data.head())

print("Amount of colomns = ", ODI_data.shape[1])
print("Amount of answers = ", ODI_data.shape[0])


def chocolate_gender(df):    
    """
    Looks at how people from different gender look towards chocolate
    """
    sns.catplot(x="Gender", hue="Chocolate", kind="count", data=df, legend=True)
    
    plt.xlabel("", fontsize=14)
    plt.ylabel("", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("chocolate_gender.pdf")
    plt.show()


def stress_esteam(df):
    """
    Looks at the relation between stress and the amount of money one would think
    he/she would earn when doing DM
    """


    print(stats.spearmanr(df["Stress_c"], df["Self esteem_c"]))
    sns.scatterplot(data=df, x="Stress_c", y="Self esteem_c")
    plt.xlabel("Stress", fontsize=14)
    plt.ylabel("Self esteem", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("stress_esteem.pdf")
    plt.show()
def programme_count(df):
    sns.catplot(x="Programme_c", kind="count", data=df)
    plt.show()

def stress_Msc(df):

    sns.catplot(x="Programme_c", y="Stress_c", data=df)
    plt.show()

def Age_stress(df):
    sns.scatterplot(data=df, x="Stress_c", y="Age")
    plt.show()


def stress_check(df, course):
    """
    Checks the difference in stress between a group who did do a course
    and a group who did not do that course
    """

    sns.catplot(x=course, y="Stress_c", data=df)
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
    

    # Data cleaners

    df = Data_cleaner.stress_cleaner(df)
    # df = Data_cleaner.classify_numericals(df, "Stress_c")
    # Data_cleaner.birth_date_cleaner(df)
    # df = Data_cleaner.calc_age(df)
    # Age_stress(df)

    df = Data_cleaner.programme_cleaner(df)

    df = Data_cleaner.se_cleaner(df)

    # for i in ["ML", "IR", "Stat", "DB"]:
    #     df = Data_cleaner.categorical(df, i)
    
    # df = Data_cleaner.categorical(df, "Programme_c", course=False)
    
    # df = df.dropna()
    

    # Plots

    # stress_check(df, "IR")
    stress_esteam(df)
    # chocolate_gender(df)

  
    
    # print(df["Programme_c"].head(100))
    # print(df["Programme"].head(100))
    # sns.catplot(x="Programme_c", kind="count", data=df)
    # plt.show()

    stress_Msc(df)


    for i in df.columns:
        print(i)


    # Categorisations

    features = ["ML,no" , "ML,yes" ,"IR,no", "IR,yes" ,"IR,uk" ,"Stat,no", "Stat,yes", "Stat,uk" ,"DB,no" , "DB,yes" ,"DB,uk"]
    features = df["Programme_c"].unique()
    y = "Stress_c"


    Categorisations.tree(df, features, y)
    Categorisations.forest(df, features, y)
    Categorisations.bayes(df, features, y)
    

    

 
