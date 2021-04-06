#%%
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

# ODI_data = pd.read_csv("Ass 1 - basic/Data/ODI-2021.csv")
ODI_data = pd.read_csv("Data/ODI-2021.csv")

print(ODI_data.head())

#%%
""" Properties of dataset """

# Colomn questions 
Q = 0
for col in ODI_data.columns:
    col_nunique = ODI_data[col].nunique()
    print("Q"+str(Q)+":",col)
    print("Num unique values:", col_nunique,"\n")
    Q += 1

# Q1,8,9,11-16 are open answerable questions
# Others are categorical multiple choice questions

print("Amount of colomns = ", ODI_data.shape[1])
print("Amount of answers = ", ODI_data.shape[0])


#%%
sns.histplot(ODI_data["What is your stress level (0-100)?"],bins=10)

#%% 
sns.displot(ODI_data,x="What is your stress level (0-100)?", col = "What is your gender?", row = "Did you stand up?",binwidth=3, height=3, facet_kws=dict(margin_titles=True))
plt.show()

"""
TODO
- Set new colomns labels
- Filter data and make it neat
- Show outliers
- Make neat plots for the categorical colomns
- 
"""
