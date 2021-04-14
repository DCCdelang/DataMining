from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import linear_model
import math
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate

def fill_age(df):
    
    # pass
    new_df = df
    new_df = new_df.dropna()
    new_df.dropna
    X = new_df[[ "Family_Size", "Title_num", "Fare", "Pclass","Title_num", "SibSp"]]
    y = new_df["Age"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size=0.2, random_state=5)
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)

    
    predict = regr.predict(X_test)
    # print('Mean squared error: %.2f'
    #   % mean_squared_error(y_test, predict))

    intercept = regr.intercept_
    b1 = regr.coef_[0]
    b2 = regr.coef_[1]
    b3 = regr.coef_[2]
    b4 = regr.coef_[3]

 
    # print('Intercept: \n', regr.intercept_
    # )
    # print('Coefficients: \n', regr.coef_)S)

    count = -1
    for i in df["Age"]:
        count +=1
        if math.isnan(i):
            y = intercept + df.iloc[count]["Parch"] * b1 + df.iloc[count]["Binary_Sex"] * b2 + df.iloc[count]["Fare"] * b3 + df.iloc[count]["Pclass"] * b4
            if (y) > 0:
                df.loc[df.index[count], 'Age'] = y 
            else:
                df.loc[df.index[count], 'Age'] = 0
    

    return df


def Binary_Sex(df):
    df["Binary_Sex"] = df["Sex"]
    # print(df["Binary_Sex"])
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
    df["Embarked"].fillna(0)
    enc_df = pd.DataFrame(enc.fit_transform(df[["Embarked"]]).toarray())# merge with main df bridge_df on key values
    df = df.join(enc_df)
    em = ["C","Q","S", "0"]
    for i in range(4):
        df = df.rename(columns={i:em[i]})
    return df

def Embarked_2(df):
    df["Embarked"] = df["Embarked"].fillna('S')
    return df

def get_deck(df):
    floor_list = []
    for cabin in df["Cabin"]:
        if str(cabin)[0] != "n":
            floor_list.append(str(cabin)[0])
        else:
            floor_list.append(0)
    
    floor_combined = []
    for deck in floor_list:
        if deck == "A" or deck == "B" or deck == "C" or deck == "T":
            floor_combined.append("ABC")
        if deck == "D" or deck == "E":
            floor_combined.append("DE")
        if deck == "F" or deck == "G":
            floor_combined.append("FG")
        if deck == 0:
            floor_combined.append("M")

    df["Deck"] = floor_combined
    





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
    for title, sex in zip(df['Title'],df["Sex"]):
        if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            titles.append('Mr')
        elif title in ['Countess', 'Mme']:
            titles.append('Mrs')
        elif title in ['Mlle', 'Ms']:
            titles.append('Miss')
        elif title =='Dr':
            if sex =='Male':
                titles.append('Mr')
            else:
                titles.append('Mrs')
        else:
            titles.append(title)
    df['Title'] = titles
    return df

def title_num(df):
    title_num = []
    if "Title" in df:
        for title in df['Title']:
            if title == "Mr":
                title_num.append(1)
            elif title == "Miss":
                title_num.append(2)
            elif title == "Mrs":
                title_num.append(3)
            elif title == "Master":
                title_num.append(4)
    else:
        raise ValueError("No Title column!!")
    df['Title_num'] = title_num
    return df

def family_size(df):
    df['Family_Size']=df['SibSp']+df['Parch']+1
    # print(df["Family_Size"].unique())
    Fam_grouped = []
    for size in df["Family_Size"]:
        if size == 1:
            Fam_grouped.append("Alone")
        if size > 1 and size < 5:
            Fam_grouped.append("Small")
        if size >= 5 and size < 7:
            Fam_grouped.append("Medium")
        if size >= 7:
            Fam_grouped.append("Large")
    df["Family_Size_grouped"] = Fam_grouped
    return df

def is_alone(df):
    if "Family_Size" in df:
        df["Is_alone"] = 0
        df.loc[df['Family_Size'] == 1, 'Is_alone'] = 1
    else:
        raise ValueError("No family size column!!")
    return df
 
def age_class(df):
    div = [0,20,40,60]
    age_class = []
    for age in df["Age"]:
        if age >= div[0] and age <= div[1]:
            age_class.append(0)
        if age > div[1] and age <= div[2]:
            age_class.append(1)
        if age > div[2] and age <= div[3]:
            age_class.append(2)
        if age > div[3]:
            age_class.append(3)
    df["Age_div"] = age_class

    
    return df

def AgeClass(df):
    df["AgeClass"] = df["Pclass"]*df["Age_div"]
    return df