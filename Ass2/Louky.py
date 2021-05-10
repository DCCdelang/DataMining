# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 14:46:55 2021

@author: Gebruiker
"""

import pandas as pd 
import numpy as np 

import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r'C:\Users\Gebruiker\OneDrive\Computational_Science\Year1_Semester2_Block2\Data mining\Assignment1\DataMining\Ass2\Data\training_set_VU_DM.csv')
df_test = pd.read_csv(r'C:\Users\Gebruiker\OneDrive\Computational_Science\Year1_Semester2_Block2\Data mining\Assignment1\DataMining\Ass2\Data\test_set_VU_DM.csv')

def preprocessing():
    #all features appearing both in test and training set
    all_features = ['srch_id', 'date_time', 'site_id', 'visitor_location_country_id',
       'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id',
       'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool',
       'prop_location_score1', 'prop_location_score2',
       'prop_log_historical_price',  'price_usd', 'promotion_flag',
       'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window',
       'srch_adults_count', 'srch_children_count', 'srch_room_count',
       'srch_saturday_night_bool', 'srch_query_affinity_score',
       'orig_destination_distance', 'random_bool', 'comp1_rate', 'comp1_inv',
       'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv',
       'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv',
       'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv',
       'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv',
       'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv',
       'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv',
       'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv',
       'comp8_rate_percent_diff']
    #merging the test and training set for preprocessing
 
    df_whole = df[all_features].append(df_test)
    
    
    
    
def main():
    preprocessing()
    

if __name__ == '__main__':
    main()    