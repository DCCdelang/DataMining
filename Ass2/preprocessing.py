import pandas as pd
import numpy as np
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

def prop_quality_book(df):
    df["count"] = 1
    df = df.join(df.groupby(["prop_id"])["booking_bool"].sum(), on="prop_id",rsuffix="_tot")
    df = df.join(df.groupby(["prop_id"])["count"].sum(), on="prop_id",rsuffix="_tot")
    df["prob_book"] = df["booking_bool_tot"]/df["count_tot"]
    df.drop(["count"],axis=1)
    return df

def prop_quality_book_test(df_train, df_test):
    df_test["prob_book"] = 0
    df_full = pd.concat([df_train,df_test])

    df_full = df_full.join(df_full.groupby(["prop_id"])["prob_book"].max(), on="prop_id",rsuffix="_tot")


    df_train = df_full.iloc[:,:df_train.shape[1]]
    df_test = df_full.iloc[:,df_train.shape[1]:]
    # df_train = df_train.groupby(["prop_id"])["prob_book"].mean()
    # df_test.loc[df_test["prop_id"]==df_train["prop_id"], "prob_book"] = df_train["prob_book"]
    
    # df_test["prob_book"] = df_test["prop_id"].map(df_train.reindex("prop_id")["prob_book"])

    return df_test

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

def drop_nan_columns(df, threshhold=0.1):

    df1 = df.dropna(axis=1, thresh= threshhold * df.shape[0])

    for i in list(df.columns):
        if i not in (df1.columns):
            print(i)

    return df1


if __name__ == "__main__":
    start = time.time()
    df = pd.read_csv('Ass2/Data/training_set_VU_DM.csv')
    
    print(time.time() - start)

    df_test = pd.read_csv('Ass2/Data/test_set_VU_DM.csv')
    print(time.time() - start)

    df_test = prop_quality_book_test(df, df_test)
    
    print(time.time() - start)
    print(df_test.head(100))

    exit()
    drop_nan_columns(df, threshhold=0.05)

    print(time.time() - start)
    extract_date_time(df)

    print(time.time() - start)
    price_per_day(df)

    print(time.time() - start)
    exp_historical_price_dif(df)

    print(time.time() - start)
    log_historical_price_dif(df)

    print(time.time() - start)
    starrating_diff(df)

    print(time.time() - start)
    prop_quality_book(df)

    print(time.time() - start)
    averages_per_prop(df)

    print(time.time() - start)
    std_per_prop(df)

    print(time.time() - start)
    median_per_prop(df)

    print(time.time() - start)

    # df.to_csv('Data/preprocessed.csv')