import pandas as pd
import numpy as np
from numba import jit
import time
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Small subset
# df = pd.read_csv("Ass2/Data/training_set_VU_DM.csv")
# df.head(10000).to_csv("training_head.csv")

df = pd.read_csv("Ass2/training_head.csv")
# print(df.columns)
# print(df.nunique())

"""
TODO:

Variables with missing 
"""
# print(df.dtypes)

# Filters rows with booking boolean is 1
def booking_Filter(df):
    df = df.loc[df["booking_bool"] == 1]
    return df

# Filters rows with clicking boolean is 1
def clicking_Filter(df):
    df = df.loc[df["click_bool"] == 1]
    return df

# Plot correlation plot full df
def plot_corr(df):
    _, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax)
    plt.show()

# Make extra colomns time and hour, plot possibility for hour distribution
def extract_date_time(df, plot = False):
    df['date'] = pd.to_datetime(df['date_time'], format="%Y-%d-%m %H:%M:%S",infer_datetime_format=True)

    df["time"] = df["date"].dt.time
    df['hour'] = df['date'].dt.hour

    if plot == True:
        sns.histplot(df["hour"],bins=24)
        plt.show()
    return df

# Scatterplot for prop location colomns, important features!
def prop_loc_plot(df):
    sns.scatterplot(x=df["prop_location_score1"], y=df["prop_location_score2"])
    plt.show()

# Plot position on list with respect to clicking boolean
def position_click(df):
    df_book = booking_Filter(df)
    df_click = clicking_Filter(df)

    plt.hist([df_book["position"],df_click["position"]],color=["r","b"], label=["Booking","Clicking"],alpha=0.5,bins=37)
    plt.legend()
    plt.show()
    print(set(df_click["position"]))

def summary(df):
    print(df.describe)
    print(df.nunique())

# Results per query
def hotels_per_id(df):
    group_id = df.groupby(["srch_id"]).size()
    print("Hotels per srch_id \nMean:", group_id.mean(),"\nStd:", group_id.std())

# Gross price divided by days of stay
def price_per_day(df):
    # df = booking_Filter(df).copy()
    df["gross_bookings_per_day"] = df["gross_bookings_usd"]/df["srch_length_of_stay"]

    df["price_per_day"] = df["price_usd"]/df["srch_length_of_stay"]
    return df

summary(df)    

# df = price_per_day(df)
# print(df.head())
