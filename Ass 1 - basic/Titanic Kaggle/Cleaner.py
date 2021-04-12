from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from sklearn import linear_model
import math

def fill_age(df):
    new_df = df
    new_df = new_df.dropna()
    new_df.dropna
    X = new_df[["Fare", "Pclass", "SibSp"]]
    y = new_df["Age"]

    regr = linear_model.LinearRegression()
    regr.fit(X, y)

    intercept = regr.intercept_
    fare_co = regr.coef_[0]
    pclass_co = regr.coef_[1]
    sib_co = regr.coef_[2]
    print('Intercept: \n', regr.intercept_)
    print('Coefficients: \n', regr.coef_)


    for count, i in enumerate(df["Age"]):
        if math.isnan(i):
            df.loc[df.index[count], 'Age'] = intercept + df.loc[count]["Fare"] * fare_co + df.loc[count]["Pclass"] * pclass_co + df.loc[count]["SibSp"] * sib_co
    return df


def Binary_Sex(df):
    df["Binary_Sex"] = df["Sex"]
    print(df["Binary_Sex"])
    df["Binary_Sex"] = df["Binary_Sex"].replace(["female", "male"], [1, 2])
    return df

def Class(df):
    enc = OneHotEncoder(handle_unknown='ignore')# passing bridge-types-cat column (label encoded values of bridge_types)

    enc_df = pd.DataFrame(enc.fit_transform(df[["Pclass"]]).toarray())# merge with main df bridge_df on key values
    df = df.join(enc_df)
    for i in range(3):
        df = df.rename(columns={i:f"C{i+1}"})
    return df

def Binary_cabin(df):
    df["Cabin_Binary"] = df["Cabin"]
    for i in df["Cabin_Binary"]:

        if type(i) == str:
            df["Cabin_Binary"] = df["Cabin_Binary"].replace(i,1)
        else:
            df["Cabin_Binary"] = df["Cabin_Binary"].replace(i,0)
    return df 

def SexClass(df):
    df["SexClass"] = df["Pclass"]*df["Binary_Sex"]
    return df

def Embarked(df):
    enc = OneHotEncoder(handle_unknown='ignore')# passing bridge-types-cat column (label encoded values of bridge_types)

    enc_df = pd.DataFrame(enc.fit_transform(df[["Embarked"]]).toarray())# merge with main df bridge_df on key values
    df = df.join(enc_df)
    em = ["C","Q","S", "0"]
    for i in range(4):
        df = df.rename(columns={i:em[i]})


    return df