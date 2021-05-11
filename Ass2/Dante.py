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



""" Function call """
# df = pd.read_csv("Ass2/Data/train_data.csv")

start = time.time()
# df = pd.read_csv("Ass2/Data/training_set_VU_DM.csv")


df_train = pd.read_csv('Ass2/Data/training_head.csv')
df_train.head(1000).to_csv("Ass2/Data/training_head_s.csv",index=False)

df_test = pd.read_csv('Ass2/Data/training_head.csv')
df_test.head(1000).to_csv("Ass2/Data/test_head_s.csv",index=False)

exit()

start2 = time.time()
print("loading:",start2-start)
# df = log_historical_price_dif(df)

# df = averages_per_prop(df)
end = time.time()

print("feature:",end-start2)

# position_click(df)
# prop_loc_plot(df)
# exp_historical_price_dif(df)
# plot_corr(df)
# print(df.head())
# prop_quality(df)

# df["log_price_usd"] = df["log_price_usd"].clip(2,8)
# sns.histplot(df["log_price_usd"])
# plt.show()
# print(df.head())