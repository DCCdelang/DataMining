# -*- coding: utf-8 -*-
"""
Created on Mon May 10 12:56:29 2021

@author: Gebruiker
"""
import pandas as pd 
import numpy as np 

import seaborn as sns
import matplotlib.pyplot as plt

plt.matplotlib.rcParams.update({'font.size': 18})

df = pd.read_csv(r'C:\Users\Gebruiker\OneDrive\Computational_Science\Year1_Semester2_Block2\Data mining\Assignment1\DataMining\Ass2\Data\training_set_VU_DM.csv')
df_test = pd.read_csv(r'C:\Users\Gebruiker\OneDrive\Computational_Science\Year1_Semester2_Block2\Data mining\Assignment1\DataMining\Ass2\Data\test_set_VU_DM.csv')

 
def analyze():
    df_clicked =df[df['click_bool'] == 1]
    df_new1 = df_clicked[['prop_id', 'random_bool', 'booking_bool', 'position']]


    """Plotting the position bias plot - Random order """
    print('Position bias')
    print("Since Expedia’s algorithm is performing better than random, there is a clear correspondence between the position of a hotel on the list of results,and the probability that the hotel was clicked on or booked. What is not as clear, is the amount of bias introduced by the positioning, even if the hotels are presented in random order.")
    # creating subplots
    plt.figure(figsize=(15, 6))
    sns.countplot(x = 'position', data = df_new1[df_new1.random_bool == 1], hue = 'booking_bool')
    plt.title('Clicked and booked: Random order')
    plt.tight_layout()
    #plt.savefig('Position_bias_random.pdf')
    plt.show()
    
    """PLotting position bias plot - Expedia order """
    plt.figure(figsize=(15, 6))
    sns.countplot(x = 'position', data = df[(df.random_bool == 0) & (df.click_bool == 1)], hue = 'booking_bool')
    plt.title('Clicked and booked: Expedia order')
    plt.tight_layout()
    #plt.savefig('Position bias expedia.pdf')    
    plt.show()
    
    """Plotting Star and price difference"""
    df['price_diff'] = np.abs(df['price_usd'] - df['visitor_hist_adr_usd'])
    df['star_diff'] = np.abs(df['prop_starrating'] - df['visitor_hist_starrating'])
    df['book_click'] = df['booking_bool'] + df['click_bool']
    print("Price difference")
    sns.barplot(x = 'book_click', y = 'price_diff', data = df)
    plt.xlabel('Booked or clicked')
    plt.ylabel('Difference in price')
    plt.tight_layout()
    #plt.savefig('Price_diff.pdf')
    plt.show()
    print("Star difference")
    sns.barplot(x = 'book_click', y = 'star_diff', data = df)
    plt.xlabel('Booked or clicked')
    plt.ylabel('Difference in star rating')
    plt.tight_layout()
    #plt.savefig('Star_diff.pdf')
    plt.show()
  
def correlation_plot():
    search_criteria = df[['srch_length_of_stay', 'srch_booking_window','srch_adults_count', 'srch_children_count', 'srch_room_count',
       'srch_saturday_night_bool']]
    Hotel_stats = df[ [ 'prop_starrating', 'prop_review_score', 'prop_brand_bool',
       'prop_location_score1', 'prop_location_score2',
       'prop_log_historical_price',  'promotion_flag']]
   
    corr = search_criteria.corr()
    corr2 =  Hotel_stats.corr()
    
    print("Search criteria Correlations")
    corr.style.background_gradient(cmap='coolwarm').set_precision(2)
    print('Hotel charachteristics correlations')
    corr2.style.background_gradient(cmap='coolwarm').set_precision(2)
    print("Correlations with click and booking bools")
    corr3 = df.corr()
    corr3[['click_bool', 'booking_bool']].style.background_gradient(cmap='coolwarm').set_precision(2)

    
  
def main():
    analyze()
    correlation_plot()


if __name__ == "__main__":
    main()