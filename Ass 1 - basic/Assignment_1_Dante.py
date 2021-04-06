#%%
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import Data_cleaner
import nltk
# nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
import flair
flair_sentiment = flair.models.TextClassifier.load('en-sentiment')

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

#%%
# Import data
df = ODI_data

# Make new collumn names
new_cols = ["Time", "Programme", "ML", "IR", "Stat", "DB","Gender","Chocolate","Birthday","Neighbours", "Stand up", "Stress", "Self esteem", "RN", "Bedtime","GD1", "GD2"]
df = Data_cleaner.rename_collumns(df,new_cols)

#%%

# Adding some simple sentiment analyses on both open questions GD1 and 2

def predict_flair(sentence):
    """ Predict the sentiment of a sentence """
    if sentence == "":
        return 0
    text = flair.data.Sentence(sentence)
    # stacked_embeddings.embed(text)
    flair_sentiment.predict(text)
    value = text.labels[0].to_dict()['value'] 
    if value == 'POSITIVE':
        result = text.to_dict()['labels'][0]['confidence']
    else:
        result = -(text.to_dict()['labels'][0]['confidence'])
    return round(result, 3)

NLTK1 = []
FLAIR1 = []
for sentence in df["GD1"]:
    NLTK1.append(list(sid.polarity_scores(sentence).values())[-1])
    FLAIR1.append(predict_flair(sentence))

NLTK2 = []
FLAIR2 = []
for sentence in df["GD2"]:
    NLTK2.append(list(sid.polarity_scores(sentence).values())[-1])
    FLAIR2.append(predict_flair(sentence))

df["GD1-NLTK1"] = NLTK1
df["GD1-FLAIR1"] = FLAIR1
df["GD1-NLTK2"] = NLTK2
df["GD1-FLAIR2"] = FLAIR2

df.head()


#%%
# Bedtime preprocessing

testtime = "23:00"

import re
r = re.compile('.*:.*')

Formatted = []
for time in df["Bedtime"]:
    time_int  = re.findall(r'[0-9]+', time)
    # if r.match(time) is not None:
    #     print(time)
    #     time_int = [int(s) for s in time.split() if s.isdigit()]
    Formatted.append(time_int)

print(Formatted)
# %%
