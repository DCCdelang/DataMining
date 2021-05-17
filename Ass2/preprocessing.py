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
    df["price_per_day"] = df["price_usd"]/df["srch_length_of_stay"]
    return df

# Convert to usd prices and difference
def exp_historical_price_dif(df):
    df["historical_price"] = np.exp(df["prop_log_historical_price_avg_prop"])
    df["hist_price_dif_avg"] = df["historical_price"] - df["price_per_day_avg_prop"]
    return df

def starrating_diff(df):
    df["starrating_diff"]=np.abs(df["visitor_hist_starrating"]-df["prop_starrating"])
    return df

def prob_quality_click(df):
    df["count"] = 1
    df = df.join(df.groupby(["prop_id"])["click_bool"].sum(), on="prop_id",rsuffix="_tot")
    df = df.join(df.groupby(["prop_id"])["count"].sum(), on="prop_id",rsuffix="_tot")
    df["prob_click"] = df["click_bool_tot"]/df["count_tot"]
    df = df.drop(["count"],axis=1)
    df = df.drop(["click_bool_tot"],axis=1)
    df = df.drop(["count_tot"],axis=1)
    return df

def prob_quality_click_test(df_train, df_test):
    df_test["prob_click"] = 0

    df_full = pd.concat([df_train,df_test])

    df_full = df_full.join(df_full.groupby(["prop_id"])["prob_click"].max(), on="prop_id",rsuffix="_extend")

    df_test["prob_click"]  = df_full[df_full["train_bool"]==0]["prob_click_extend"]
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
    df_full = pd.concat([df_train,df_test])

    df_full = df_full.join(df_full.groupby(["prop_id"])["prob_book"].max(), on="prop_id",rsuffix="_extend")

    df_test["prob_book"]  = df_full[df_full["train_bool"]==0]["prob_book_extend"]
    return df_test

def position_average(df_train,df_test):
    df_train_ranked = df_train.loc[df_train["random_bool"] == 0]
    df_train_ranked = df_train_ranked.join(df_train_ranked.groupby(["prop_id"])["position"].mean(),on="prop_id",rsuffix="_mean")

    df_test["position_mean"] = 0
    df_test_ranked = df_test.loc[df_test["random_bool"] == 0]

    df_train_ranked = df_train_ranked[["prop_id","train_bool","position_mean"]]
    df_test_ranked = df_test_ranked[["prop_id","train_bool","position_mean"]]

    df_ranked = pd.concat([df_train_ranked, df_test_ranked])
    df_ranked = df_ranked.join(df_ranked.groupby(["prop_id"])["position_mean"].max(),on="prop_id",rsuffix="_ex")

    df_ranked = df_ranked.drop(["position_mean"],axis=1)

    df_ranked = pd.concat([df_train, df_test,df_ranked])
    df = df_ranked.join(df_ranked.groupby(["prop_id"])["position_mean_ex"].max(),on="prop_id",rsuffix="tend")    

    df = df.drop(["position_mean_ex","position_mean"],axis=1)
    df = df[df["srch_id"].notna()]

    df_train = df[df["train_bool"]==1]
    df_test = df[df["train_bool"]==0]

    df_test = df_test.drop(["gross_bookings_usd","click_bool","booking_bool"],axis=1)

    df_train.loc[df_train.random_bool == 1, 'position_mean_extend'] = 0
    df_test.loc[df_test.random_bool == 1, 'position_mean_extend'] = 0

    return df_train, df_test

def position_average_simple(df_train,df_test):
    print(df_test.shape)
    print(df_train.shape)
    df_train_ranked = df_train.loc[df_train["random_bool"] == 0]
    df_train_ranked = df_train_ranked.join(df_train_ranked.groupby(["prop_id"])["position"].mean(),on="prop_id",rsuffix="_mean")

    # df_test["position_mean"] = 0
    # df_test_ranked = df_test.loc[df_test["random_bool"] == 0]

    # df_train_ranked = df_train_ranked[["prop_id","train_bool","position_mean"]]
    # df_test_ranked = df_test_ranked[["prop_id","train_bool","position_mean"]]

    df_ranked = pd.concat([df_train, df_train_ranked, df_test])
    df_ranked = df_ranked.join(df_ranked.groupby(["prop_id"])["position_mean"].max(),on="prop_id",rsuffix="_ex")

    df = df_ranked.drop(["position_mean"],axis=1)

    # df_ranked = pd.concat([df_train, df_test,df_ranked])
    # df = df_ranked.join(df_ranked.groupby(["prop_id"])["position_mean_ex"].max(),on="prop_id",rsuffix="tend")    

    # df = df.drop(["position_mean_ex","position_mean"],axis=1)

    df_train = df[df["train_bool"]==1]
    df_test = df[df["train_bool"]==0]

    df_test = df_test.drop(["position","gross_bookings_usd","click_bool","booking_bool"],axis=1)

    df_train.loc[df_train.random_bool == 1, 'position_mean_extend'] = 0
    df_test.loc[df_test.random_bool == 1, 'position_mean_extend'] = 0

    return df_train, df_test

# Function to average out numerical values per property, can be done in combination with test set. Should be done at beginning?!
def averages_per_prop(df_train, df_test):
    df = pd.concat([df_train,df_test])
    
    df["count"] = 1
    df = df.join(df.groupby(["prop_id"])["prop_starrating"].mean(), on="prop_id",rsuffix="_avg_prop")

    df = df.join(df.groupby(["prop_id"])["prop_review_score"].mean(), on="prop_id",rsuffix="_avg_prop")

    df = df.join(df.groupby(["prop_id"])["prop_location_score1"].mean(), on="prop_id",rsuffix="_avg_prop")

    df = df.join(df.groupby(["prop_id"])["prop_location_score2"].mean(), on="prop_id",rsuffix="_avg_prop")

    df = df.join(df.groupby(["prop_id"])["prop_log_historical_price"].mean(), on="prop_id",rsuffix="_avg_prop")

    df = df.join(df.groupby(["prop_id"])["price_per_day"].mean(), on="prop_id",rsuffix="_avg_prop")
    
    df = df.drop(["count"],axis=1)
    df_train = df[df["train_bool"]==1]
    df_test = df[df["train_bool"]==0]
    df_test = df_test.drop(["gross_bookings_usd","click_bool","booking_bool"],axis=1)
    return df_train,df_test

def averages_per_srch_id(df_train, df_test):
    df = pd.concat([df_train,df_test])
    
    df["count"] = 1
    df = df.join(df.groupby(["srch_id"])["prop_starrating"].mean(), on="srch_id",rsuffix="_avg_srch")

    df = df.join(df.groupby(["srch_id"])["prop_review_score"].mean(), on="srch_id",rsuffix="_avg_srch")

    df = df.join(df.groupby(["srch_id"])["prop_location_score1"].mean(), on="srch_id",rsuffix="_avg_srch")

    df = df.join(df.groupby(["srch_id"])["prop_location_score2"].mean(), on="srch_id",rsuffix="_avg_srch")

    df = df.join(df.groupby(["srch_id"])["prop_log_historical_price"].mean(), on="srch_id",rsuffix="_avg_srch")

    df = df.join(df.groupby(["srch_id"])["price_per_day"].mean(), on="srch_id",rsuffix="_avg_srch")
    
    df = df.drop(["count"],axis=1)
    df_train = df[df["train_bool"]==1]
    df_test = df[df["train_bool"]==0]
    df_test = df_test.drop(["gross_bookings_usd","click_bool","booking_bool"],axis=1)
    return df_train,df_test

def averages_per_destin(df_train, df_test):
    df = pd.concat([df_train,df_test])
    
    df["count"] = 1
    df = df.join(df.groupby(["srch_destination_id"])["prop_starrating"].mean(), on="srch_destination_id",rsuffix="_avg_dest")

    df = df.join(df.groupby(["srch_destination_id"])["prop_review_score"].mean(), on="srch_destination_id",rsuffix="_avg_dest")

    df = df.join(df.groupby(["srch_destination_id"])["prop_location_score1"].mean(), on="srch_destination_id",rsuffix="_avg_dest")

    df = df.join(df.groupby(["srch_destination_id"])["prop_location_score2"].mean(), on="srch_destination_id",rsuffix="_avg_dest")

    df = df.join(df.groupby(["srch_destination_id"])["prop_log_historical_price"].mean(), on="srch_destination_id",rsuffix="_avg_dest")

    df = df.join(df.groupby(["srch_destination_id"])["price_per_day"].mean(), on="srch_destination_id",rsuffix="_avg_dest")
    
    df = df.drop(["count"],axis=1)
    df_train = df[df["train_bool"]==1]
    df_test = df[df["train_bool"]==0]
    df_test = df_test.drop(["gross_bookings_usd","click_bool","booking_bool"],axis=1)
    return df_train,df_test

def std_per_prop(df_train, df_test):
    df = pd.concat([df_train,df_test])

    df["count"] = 1
    
    df = df.join(df.groupby(["prop_id"])["prop_starrating"].std(), on="prop_id",rsuffix="_std_prop")

    df = df.join(df.groupby(["prop_id"])["prop_review_score"].std(), on="prop_id",rsuffix="_std_prop")

    df = df.join(df.groupby(["prop_id"])["prop_location_score1"].std(), on="prop_id",rsuffix="_std_prop")

    df = df.join(df.groupby(["prop_id"])["prop_location_score2"].std(), on="prop_id",rsuffix="_std_prop")

    df = df.join(df.groupby(["prop_id"])["prop_log_historical_price"].std(), on="prop_id",rsuffix="_std_prop")

    df = df.join(df.groupby(["prop_id"])["price_per_day"].std(), on="prop_id",rsuffix="_std_prop")

    df = df.drop(["count"],axis=1)
    df_train = df[df["train_bool"]==1]
    df_test = df[df["train_bool"]==0]
    df_test = df_test.drop(["gross_bookings_usd","click_bool","booking_bool"],axis=1)
    return df_train,df_test

def median_per_prop(df_train, df_test):
    df = pd.concat([df_train,df_test])

    df["count"] = 1
    df = df.join(df.groupby(["prop_id"])["prop_starrating"].median(), on="prop_id",rsuffix="_median_prop")

    df = df.join(df.groupby(["prop_id"])["prop_review_score"].median(), on="prop_id",rsuffix="_median_prop")

    df = df.join(df.groupby(["prop_id"])["prop_location_score1"].median(), on="prop_id",rsuffix="_median_prop")

    df = df.join(df.groupby(["prop_id"])["prop_location_score2"].median(), on="prop_id",rsuffix="_median_prop")

    df = df.join(df.groupby(["prop_id"])["prop_log_historical_price"].median(), on="prop_id",rsuffix="_median_prop")

    df = df.join(df.groupby(["prop_id"])["price_per_day"].median(), on="prop_id",rsuffix="_median_prop")

    df = df.drop(["count"],axis=1)
    df_train = df[df["train_bool"]==1]
    df_test = df[df["train_bool"]==0]
    df_test = df_test.drop(["gross_bookings_usd","click_bool","booking_bool"],axis=1)
    return df_train,df_test

def drop_nan_columns(df, threshhold=0.1):
    df1 = df.dropna(axis=1, thresh= threshhold * df.shape[0])
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

Functie aanmaken die training set en test set samenvoegt en vervolgens gemiddelde etc.
pakt en vervolgens weer uitelkaar haalt.

Kijken naar negatieve waardes (en niet alleen clicked_data)
"""


if __name__ == "__main__":
    start = time.time()

    df_train = pd.read_csv('Data/training_set_VU_DM.csv')
    df_test = pd.read_csv('Data/test_set_VU_DM.csv')
   
    # df_train = pd.read_csv('Data/training_head.csv')
    # df_test = pd.read_csv('Data/test_head.csv')

    # df_train = df_train.drop(["Unnamed: 0"],axis = 1)
    # df_test = df_test.drop(["Unnamed: 0"],axis = 1)

    # print("OG Shape train",df_train.shape)
    # print("OG Shape test",df_test.shape)

    df_train["train_bool"] = 1
    df_test["train_bool"] = 0

    # df_train = drop_nan_columns(df_train, threshhold=0.7)
    # df_test = drop_nan_columns(df_test, threshhold=0.7)

    df_train = df_train.drop(["date_time"],axis=1)
    df_test = df_test.drop(["date_time"],axis=1)

    """WERKEN ALLEBEI NOG NIET GVD"""
    df_train,df_test = position_average(df_train,df_test)
    # df_train,df_test = position_average_simple(df_train,df_test)
    print('1')

    print(df_test.shape)
    print(df_train.shape)

    df_train = prob_quality_book(df_train)
    df_test = prob_quality_book_test(df_train, df_test)
    print('2')

    print(df_test.shape)
    print(df_train.shape)

    df_train = prob_quality_click(df_train)
    df_test = prob_quality_click_test(df_train, df_test)
    print('3')

    print(df_test.shape)
    print(df_train.shape)

    df_train = price_per_day(df_train)
    df_test = price_per_day(df_test)
    print('4.1')

    print(df_test.shape)
    print(df_train.shape)

    df_train,df_test = averages_per_prop(df_train, df_test)
    df_train,df_test = std_per_prop(df_train, df_test)
    df_train,df_test = median_per_prop(df_train, df_test)
    
    print("4.2")
    print(df_test.shape)
    print(df_train.shape)
    
    df_train,df_test = averages_per_srch_id(df_train, df_test)
    
    print("4.2")
    print(df_test.shape)
    print(df_train.shape)

    df_train,df_test = averages_per_destin(df_train, df_test)

    print('5')

    print(df_test.shape)
    print(df_train.shape)

    df_train = exp_historical_price_dif(df_train)
    df_test = exp_historical_price_dif(df_test)

    print('6')

    print(df_test.shape)
    print(df_train.shape)

    df_train = starrating_diff(df_train)
    df_test = starrating_diff(df_test)

    print('7')

    print(df_test.shape)
    print(df_train.shape)

    drop_nan_columns(df_test, threshhold=0.05)
    drop_nan_columns(df_train, threshhold=0.05)
    df_train = df_train.round(2)
    df_test = df_test.round(2)

    print('8')

    print(df_test.shape)
    print(df_train.shape)

    print('1')
    df_train.to_csv('Data/prepro_train2.csv', index=False)
    print('2')
    df_test.to_csv('Data/prepro_test2.csv', index=False)
    print('3')
