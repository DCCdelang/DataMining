# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 17:50:55 2021

@author: Gebruiker
"""
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


#main libraries
import os
import re
import pickle
import numpy as np
import pandas as pd

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly 
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.offline import iplot, init_notebook_mode
# cufflinks as cf

train_df = pd.read_csv("Data/train.csv")
test_df  = pd.read_csv("Data/test.csv")

# some_plots(Titan_data)

def more_plots(full_df ):
    print(full_df.columns)
    #the shape of the data
    print('This data contains {} rows and {} columns splited into train/test datasets with ratio {}'.\
      format(full_df.shape[0],full_df.shape[1],round((test_df.shape[0]/train_df.shape[0])*100,2)))
        
    #describe data
    train_df[train_df.select_dtypes(exclude='object').columns].drop('PassengerId',axis=1).describe().\
        style.background_gradient(axis=1,cmap=sns.light_palette('green', as_cmap=True))
   
""" Transformation on original dataset """

def main():

    #join all the data together
    full_df = pd.concat([train_df,test_df])

    #make a copy of the original data
    train_df_orig = train_df.copy()
    test_df_orig = test_df.copy()
    
    more_plots(full_df)
    
if __name__ == "__main__":
    main()


