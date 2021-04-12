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
            if intercept + df.loc[count]["Fare"] * fare_co + df.loc[count]["Pclass"] * pclass_co + df.loc[count]["SibSp"] * sib_co > 0:
                df.loc[df.index[count], 'Age'] = intercept + df.loc[count]["Fare"] * fare_co + df.loc[count]["Pclass"] * pclass_co + df.loc[count]["SibSp"] * sib_co
            else:
                df.loc[df.index[count], 'Age'] = 0
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

def get_deck(df):
    floor_list = []
    for cabin in df["Cabin"]:
        if str(cabin)[0] != "n":
            floor_list.append(str(cabin)[0])
        else:
            floor_list.append(np.nan)
    df["deck"] = floor_list

def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if big_string.find(substring) != -1:
            return substring
    return np.nan

def replace_titles(df):
    title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                    'Don', 'Jonkheer']

    df['Title']=df['Name'].map(lambda x: substrings_in_string(x, title_list))

    titles = []
    for title in df['Title']:
        if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            return 'Mr'
        elif title in ['Countess', 'Mme']:
            return 'Mrs'
        elif title in ['Mlle', 'Ms']:
            return 'Miss'
        elif title =='Dr':
            if df['Sex']=='Male':
                return 'Mr'
            else:
                return 'Mrs'
        else:
            titles.append(title)

    df['Title'] = titles
    return df

def family_size(df):
    df['Family_Size']=df['SibSp']+df['Parch']
    return df

def is_alone(df):
    if "Family_Size" in df:
        df["Is_alone"] = 0
        df.loc[df['Family_Size'] == 1, 'Is_alone'] = 1
    else:
        print("No family size colomn!")
    return df
 
def age_class(df):
    div = [0,21,35,55]
    age_class = []
    for age in df["Age"]:
        if age >= div[0] and age <= div[1]:
            age_class.append("0")
        if age > div[1] and age <= div[2]:
            age_class.append("1")
        if age > div[2] and age <= div[3]:
            age_class.append("2")
        if age > div[3]:
            age_class.append("3")
    df["Age_div"] = age_class
    return df