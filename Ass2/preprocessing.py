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
    # df["gross_bookings_per_day"] = df["gross_bookings_usd"]/df["srch_length_of_stay"]
    df["price_per_day"] = df["price_usd"]/df["srch_length_of_stay"]
    return df

# Convert to usd prices and difference
def exp_historical_price_dif(df):
    df["historical_price"] = np.exp(df["prop_log_historical_price"])
    df["hist_price_dif"] = df["historical_price"] - df["price_usd"]
    return df

# Convert to log prices and difference
# def log_historical_price_dif(df):
#     df["log_price_usd"] = np.log(df["price_usd"])
#     df["log_hist_price_dif"] = df["prop_log_historical_price"] - df["log_price_usd"]
#     return df

def starrating_diff(df):
    df["starrating_diff"]=np.abs(df["visitor_hist_starrating"]-df["prop_starrating"])
    return df

def prob_quality_click(df):
    df["count"] = 1
    df = df.join(df.groupby(["prop_id"])["click_bool"].sum(), on="prop_id",rsuffix="_tot")
    df = df.join(df.groupby(["prop_id"])["count"].sum(), on="prop_id",rsuffix="_tot")
    # print(df["click_bool_tot"].unique())
    # print(df["count_tot"].unique())
    df["prob_book"] = df["click_bool_tot"]/df["count_tot"]
    df = df.drop(["count"],axis=1)
    df = df.drop(["click_bool_tot"],axis=1)
    df = df.drop(["count_tot"],axis=1)
    return df

def prob_quality_click_test(df_train, df_test):
    df_test["prob_book"] = 0
    train_sub = df_train[["prop_id","prob_book"]]
    test_sub = df_test[["prop_id","prob_book"]]

    df_full = pd.concat([train_sub,test_sub])

    df_full = df_full.join(df_full.groupby(["prop_id"])["prob_book"].max(), on="prop_id",rsuffix="_extend")

    df_test["prob_book"] = df_full["prob_book_extend"].tail(df_test.shape[0])

    return df_test

def prob_quality_book(df):
    df["count"] = 1
    df = df.join(df.groupby(["prop_id"])["booking_bool"].sum(), on="prop_id",rsuffix="_tot")
    df = df.join(df.groupby(["prop_id"])["count"].sum(), on="prop_id",rsuffix="_tot")
    df["prob_book"] = df["booking_bool_tot"]/df["count_tot"]
    df = df.drop(["count"],axis=1)
    df = df.drop(["booking_bool_tot"],axis=1)
    df = df.drop(["count_tot"],axis=1)
    return df

def prob_quality_book_test(df_train, df_test):
    df_test["prob_book"] = 0
    train_sub = df_train[["prop_id","prob_book"]]
    test_sub = df_test[["prop_id","prob_book"]]

    df_full = pd.concat([train_sub,test_sub])

    df_full = df_full.join(df_full.groupby(["prop_id"])["prob_book"].max(), on="prop_id",rsuffix="_extend")

    df_test["prob_book"] = df_full["prob_book_extend"].tail(df_test.shape[0])

    return df_test

def position_average(df_train,df_test):
    # Filter random positions
    print(df_train.shape[0])
    print(len(df_train.loc[df_train["random_bool"] == 0]))
    df_train_ranked = df_train.loc[df_train["random_bool"] == 0]
    df_train_ranked = df_train_ranked.join(df_train_ranked.groupby(["prop_id"])["position"].mean(),on="prop_id",rsuffix="_mean")
    
    return 


# Function to average out numerical values per property, can be done in combination with test set. Should be done at beginning?!
def averages_per_prop(df_train, df_test):
    df = pd.concat([df_train,df_test])
    
    df["count"] = 1
    df = df.join(df.groupby(["prop_id"])["prop_starrating"].mean(), on="prop_id",rsuffix="_avg")

    df = df.join(df.groupby(["prop_id"])["prop_review_score"].mean(), on="prop_id",rsuffix="_avg")

    df = df.join(df.groupby(["prop_id"])["prop_location_score1"].mean(), on="prop_id",rsuffix="_avg")

    df = df.join(df.groupby(["prop_id"])["prop_location_score2"].mean(), on="prop_id",rsuffix="_avg")

    df = df.join(df.groupby(["prop_id"])["prop_log_historical_price"].mean(), on="prop_id",rsuffix="_avg")

    df = df.join(df.groupby(["prop_id"])["price_usd"].mean(), on="prop_id",rsuffix="_avg")

    df = df.drop(["count"],axis=1)
    df_train = df.head(df_train.shape[0])
    df_test = df.head(df_test.shape[0])
    return df_train,df_test

def std_per_prop(df_train, df_test):
    df = pd.concat([df_train,df_test])

    df["count"] = 1
    
    df = df.join(df.groupby(["prop_id"])["prop_starrating"].std(), on="prop_id",rsuffix="_std")

    df = df.join(df.groupby(["prop_id"])["prop_review_score"].std(), on="prop_id",rsuffix="_std")

    df = df.join(df.groupby(["prop_id"])["prop_location_score1"].std(), on="prop_id",rsuffix="_std")

    df = df.join(df.groupby(["prop_id"])["prop_location_score2"].std(), on="prop_id",rsuffix="_std")

    df = df.join(df.groupby(["prop_id"])["prop_log_historical_price"].std(), on="prop_id",rsuffix="_std")

    df = df.join(df.groupby(["prop_id"])["price_usd"].std(), on="prop_id",rsuffix="_std")

    df = df.drop(["count"],axis=1)
    df_train = df.head(df_train.shape[0])
    df_test = df.head(df_test.shape[0])
    return df_train,df_test

def median_per_prop(df_train, df_test):
    df = pd.concat([df_train,df_test])

    df["count"] = 1
    df = df.join(df.groupby(["prop_id"])["prop_starrating"].median(), on="prop_id",rsuffix="_median")

    df = df.join(df.groupby(["prop_id"])["prop_review_score"].median(), on="prop_id",rsuffix="_median")

    df = df.join(df.groupby(["prop_id"])["prop_location_score1"].median(), on="prop_id",rsuffix="_median")

    df = df.join(df.groupby(["prop_id"])["prop_location_score2"].median(), on="prop_id",rsuffix="_median")

    df = df.join(df.groupby(["prop_id"])["prop_log_historical_price"].median(), on="prop_id",rsuffix="_median")

    df = df.join(df.groupby(["prop_id"])["price_usd"].median(), on="prop_id",rsuffix="_median")

    df = df.drop(["count"],axis=1)
    df_train = df.head(df_train.shape[0])
    df_test = df.head(df_test.shape[0])
    return df_train,df_test

def drop_nan_columns(df, threshhold=0.1):

    df1 = df.dropna(axis=1, thresh= threshhold * df.shape[0])

    for i in list(df.columns):
        if i not in (df1.columns):
            print(i)

    return df1

def drop_kkcolumns(df):
    kut_columns = ['click_bool', 'gross_bookings_usd', 'booking_bool','srch_saturday_night_bool', 'srch_query_affinity_score', 'orig_destination_distance', 'random_bool', 'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff','date_time','Unnamed: 0']
    
    df = df.drop(kut_columns, axis=1)

    return df

"""
Filosofie: Karakteristieken per property duidelijker maken zodat het algoritme
sneller/beter snapt waar een property in de lijst moet komen.
Aannames: Gemiddelde over tijd, kwaliteit property altijd hetzelfde
- Feature maken waarbij position gemiddeld genomen wordt berekend per propertie
om vervolgens over te hevelen naar de test data. 
- probabilities berekenen hoevaak een property voorkomt en hoevaak er op geklikt/
gebooked wordt. ook overhevelen naar test data.
"""


"""
Functie aanmaken die training set en test set samenvoegt en vervolgens gemiddelde etc.
pakt en vervolgens weer uitelkaar haalt.
"""

"""
Kijken naar negatieve waardes (en niet alleen clicked_data)
"""


if __name__ == "__main__":
    start = time.time()



    df_train = pd.read_csv('Ass2/Data/training_set_VU_DM.csv')
    df_test = pd.read_csv('Ass2/Data/test_set_VU_DM.csv')

    # df_train = pd.read_csv('Ass2/Data/training_head.csv')
    # df_test = pd.read_csv('Ass2/Data/test_head.csv')

    # print(time.time() - start)
    df_train = prob_quality_book(df_train)

    df_test = prob_quality_book_test(df_train, df_test)

    df_train = prob_quality_click(df_train)

    df_test = prob_quality_click_test(df_train, df_test)

    df_train = price_per_day(df_train)

    df_test = price_per_day(df_test)

    df_train,df_test = averages_per_prop(df_train, df_test)

    df_train = exp_historical_price_dif(df_train)

    df_test = exp_historical_price_dif(df_test)
    
    # drop_nan_columns(df, threshhold=0.1)
    # print(len(df_test.loc[df_test["prob_book"] == 1]))
    # print(len(df_train.loc[df_train["prob_book"] == 1]))
    # print(df_test.shape)
    # print(df_train.shape)
    # print(df_train.head(10))
    # print(df_test.head(10))

    # sns.histplot(df_test["prob_book"])
    # plt.xlim(0.00001,1)
    # plt.show()

    # sns.histplot(df_train["prob_book"])
    # plt.xlim(0.00001,1)
    # plt.show()
    
    # print(time.time() - start)
    # print(df_test.head(100))

    # exit()
    # drop_nan_columns(df, threshhold=0.05)

    # print(time.time() - start)
    # extract_date_time(df)

    # print(time.time() - start)
    # # price_per_day(df)

    # print(time.time() - start)
    # exp_historical_price_dif(df)

    # print(time.time() - start)
    # log_historical_price_dif(df)

    # print(time.time() - start)
    # starrating_diff(df)

    # print(time.time() - start)
    # # prop_quality_book(df)


    # prop_quality_book(df)

    # print(time.time() - start)
    # averages_per_prop(df)

    # print(time.time() - start)
    # std_per_prop(df)

    # print(time.time() - start)
    # median_per_prop(df)

    # print(time.time() - start)

    df_train.to_csv('Data/prepro_train.csv')
    df_test.to_csv('Data/prepro_test.csv')
