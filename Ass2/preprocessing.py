import pandas as pd
import numpy as np
from numba import jit
import time
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Make extra colomns time and hour, plot possibility for hour distribution
def extract_date_time(df, plot = False):
    df['date'] = pd.to_datetime(df['date_time'], format="%Y-%d-%m %H:%M:%S",infer_datetime_format=True)

    df["time"] = df["date"].dt.time
    df['hour'] = df['date'].dt.hour

    if plot == True:
        sns.histplot(df["hour"],bins=24)
        plt.show()
    return df

# Gross price divided by days of stay
def price_per_day(df):
    # df = booking_Filter(df).copy()
    df["gross_bookings_per_day"] = df["gross_bookings_usd"]/df["srch_length_of_stay"]
    df["price_per_day"] = df["price_usd"]/df["srch_length_of_stay"]
    return df

# Convert to usd prices and difference
def exp_historical_price_dif(df):
    df["historical_price"] = np.exp(df["prop_log_historical_price"])
    df["hist_price_dif"] = df["historical_price"] - df["price_usd"]
    return df

# Convert to log prices and difference
def log_historical_price_dif(df):
    df["log_price_usd"] = np.log(df["price_usd"])
    df["log_hist_price_dif"] = df["prop_log_historical_price"] - df["log_price_usd"]
    return df

def starrating_diff(df):
    df["starrating_diff"]=np.abs(df["visitor_hist_starrating"]-df["prop_starrating"])
    return df

def prop_quality(df):
    df["count"] = 1
    df = df.join(df.groupby(["prop_id"])["booking_bool"].sum(), on="prop_id",rsuffix="_tot")
    df = df.join(df.groupby(["prop_id"])["count"].sum(), on="prop_id",rsuffix="_tot")
    df["prob_book"] = df["booking_bool_tot"]/df["count_tot"]
    df.drop(["count"],axis=1)
    
    # print(set(df["prob_book"]))
    return df

# Function to average out numerical values per property, can be done in combination with test set. Should be done at beginning?!
def averages_per_prop(df):
    df["count"] = 1
    df = df.join(df.groupby(["prop_id"])["prop_starrating"].mean(), on="prop_id",rsuffix="_avg")

    df = df.join(df.groupby(["prop_id"])["prop_review_score"].mean(), on="prop_id",rsuffix="_avg")

    df = df.join(df.groupby(["prop_id"])["prop_location_score1"].mean(), on="prop_id",rsuffix="_avg")

    df = df.join(df.groupby(["prop_id"])["prop_location_score2"].mean(), on="prop_id",rsuffix="_avg")

    df = df.join(df.groupby(["prop_id"])["prop_log_historical_price"].mean(), on="prop_id",rsuffix="_avg")

    df = df.join(df.groupby(["prop_id"])["price_usd"].mean(), on="prop_id",rsuffix="_avg")

    df.drop(["count"],axis=1)
    return df

def std_per_prop(df):
    df["count"] = 1
    df = df.join(df.groupby(["prop_id"])["prop_starrating"].std(), on="prop_id",rsuffix="_std")

    df = df.join(df.groupby(["prop_id"])["prop_review_score"].std(), on="prop_id",rsuffix="_std")

    df = df.join(df.groupby(["prop_id"])["prop_location_score1"].std(), on="prop_id",rsuffix="_std")

    df = df.join(df.groupby(["prop_id"])["prop_location_score2"].std(), on="prop_id",rsuffix="_std")

    df = df.join(df.groupby(["prop_id"])["prop_log_historical_price"].std(), on="prop_id",rsuffix="_std")

    df = df.join(df.groupby(["prop_id"])["price_usd"].std(), on="prop_id",rsuffix="_std")

    df.drop(["count"],axis=1)
    return df

def median_per_prop(df):
    df["count"] = 1
    df = df.join(df.groupby(["prop_id"])["prop_starrating"].median(), on="prop_id",rsuffix="_median")

    df = df.join(df.groupby(["prop_id"])["prop_review_score"].median(), on="prop_id",rsuffix="_median")

    df = df.join(df.groupby(["prop_id"])["prop_location_score1"].median(), on="prop_id",rsuffix="_median")

    df = df.join(df.groupby(["prop_id"])["prop_location_score2"].median(), on="prop_id",rsuffix="_median")

    df = df.join(df.groupby(["prop_id"])["prop_log_historical_price"].median(), on="prop_id",rsuffix="_median")

    df = df.join(df.groupby(["prop_id"])["price_usd"].median(), on="prop_id",rsuffix="_median")

    df.drop(["count"],axis=1)
    return df
