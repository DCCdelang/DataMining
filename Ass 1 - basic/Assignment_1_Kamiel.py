import pandas as pd 
import numpy as np 
import math
import seaborn as sns
from Data_cleaner import Data_cleaner
import matplotlib.pyplot as plt
from scipy import stats
ODI_data = pd.read_csv("Data/ODI-2021.csv")

print(ODI_data.head())

print("Amount of colomns = ", ODI_data.shape[1])
print("Amount of answers = ", ODI_data.shape[0])


# sns.displot(ODI_data,x="What is your stress level (0-100)?", col = "What is your gender?", row = "Did you stand up?",binwidth=3, height=3, facet_kws=dict(margin_titles=True))
# plt.show()



df = ODI_data


new_cols = ["Time", "Programme", "ML", "IR", "Stat", "DB","Gender","Chocolate","Birthday","Neighbours", "Stand up", "Stress", "Self esteem", "RN", "Bedtime","GD1", "GD2"]

for i in ["Self esteem", "Stress"]:
    df = Data_cleaner(df).rename_collumns(new_cols)
    df = Data_cleaner(df).make_numeric(i)
    df = Data_cleaner(df).remove_nan()
    df = Data_cleaner(df).remove_numeric_values(i,100, 0)


# for i in df["Chocolate"]:
#     print(i)

# sns.catplot(x="Gender", hue="Chocolate", kind="count", data=df)
# plt.show()
print(stats.shapiro(df["Self esteem"]))
print(stats.shapiro(df["Stress"]))

print(stats.spearmanr(df["Stress"], df["Self esteem"]))
sns.scatterplot(data=df, x="Stress", y="Self esteem")
plt.show()
"""
TODO
- Set new colomns labels
- Filter data and make it neat
- Show outliers
- Make neat plots for the categorical colomns
- 
"""
