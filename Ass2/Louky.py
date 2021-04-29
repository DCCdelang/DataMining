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


def value_counts():
    bookings = df[['srch_id','booking_bool']].groupby('srch_id').sum()
    print('Values of booking counts:')
    print(bookings.value_counts())
    

def main():
    value_counts()
    
    

if __name__ == '__main__':
    main()    