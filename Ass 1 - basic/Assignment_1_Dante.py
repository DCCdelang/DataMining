"""
Dante de Lang
"""
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import Data_cleaner
import Assignment_1_Kamiel
import Assignment_1_Louky
import nltk
# nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
#import flair
# = flair.models.TextClassifier.load('en-sentiment')
import re

# ODI_data = pd.read_csv("Ass 1 - basic/Data/ODI-2021.csv")
ODI_data = pd.read_csv("Ass 1 - basic/Data/ODI-2021.csv")

""" Properties of dataset """

# Q1,8,9,11-16 are open answerable questions
# Others are categorical multiple choice questions

print("Amount of colomns = ", ODI_data.shape[1])
print("Amount of answers = ", ODI_data.shape[0])

"""
TODO
- Set new colomns labels
- Filter data and make it neat
- Show outliers
- Make neat plots for the categorical colomns
- 
"""

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

def add_NLP(df):
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

    df["GD1-NLTK"] = NLTK1
    df["GD1-FLAIR"] = FLAIR1
    df["GD2-NLTK"] = NLTK2
    df["GD2-FLAIR"] = FLAIR2

    return df

# Bedtime preprocessing

def bedtime_parser(df):
    Hours = []
    for time in df["Bedtime"]:
        time_int  = re.findall(r'[0-9]+', time)
        if len(time_int):
            hour_int = int(time_int[0])
            if hour_int == 10:
                hour_int = 22
            if hour_int == 11:
                hour_int = 23
            if hour_int == 12:
                hour_int = 24
            if hour_int > 24:
                Hours.append(np.nan)
            else:
                Hours.append(hour_int)
        else:
            Hours.append(np.nan)

    # print(Hours)
    # plt.hist(Hours)
    # plt.plot()
    df["Bedtime_Hour"] = Hours
    return df

if __name__ == "__main__":
    # Import data
    df = ODI_data

    # Make new collumn names
    new_cols = ["Time", "Programme", "ML", "IR", "Stat", "DB","Gender","Chocolate","Birthday","Neighbours", "Stand up", "Stress", "Self esteem", "RN", "Bedtime","GD1", "GD2"]
    df = Data_cleaner.rename_collumns(df,new_cols)
    add_NLP(df)
    bedtime_parser(df)
    Data_cleaner.stress_cleaner(df)
    Data_cleaner.se_cleaner(df)
    Data_cleaner.remove_nan(df)

    # Some plots
    plt.plot(df["Bedtime_Hour"],df["Stress_c"],".")
    plt.show()

    plt.plot(df["GD1-FLAIR"], df["GD2-FLAIR"], ".")
    plt.show()

    plt.plot((df["GD1-FLAIR"]+df["GD2-FLAIR"])/2,df["Stress_c"],".")
    plt.show()

    plt.plot((df["GD1-FLAIR"]+df["GD2-FLAIR"])/2,df["Self esteem_c"],".")
    plt.show()
    print(df.head())