import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
# from sklearn.metrics import dcg_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, AdaBoostRegressor
# from sklearn.model_selection import GridSearchCV 
# from LambdaRankNN import LambdaRankNN
from sklearn.model_selection import train_test_split
from numpy.random import RandomState
from progress.bar import Bar

rs = RandomState(42)

def add_values(df_train):
    df_train['value'] = df_train.apply(lambda row: row.click_bool + (row.booking_bool * 4), axis=1)
    return df_train

def filter_train(df_train,filter_type): # Choose Balanced, clicked
    if filter_type == "balanced":
        df_train1 = df_train.loc[df_train["click_bool"]==1]
        print(df_train1.shape)
        df_train2 = df_train[df_train["srch_id"].isin(df_train1["srch_id"])]
        size = 1        # sample size
        replace = True  # with replacement
        fn = lambda obj: obj.loc[rs.choice(obj.index, size, replace),:]
        df_train2 = df_train2.groupby('srch_id', as_index=False).apply(fn)
        print(df_train2.shape)

        df_train = pd.concat([df_train1,df_train2])

    if filter_type == "clicked":
        df_train = df_train.loc[df_train["click_bool"]==1]

    return df_train

def train_validation(df_train):
    X_train, X_test, y_train, y_test = train_test_split(df_train.drop(["value"],axis=1), df_train["value"], test_size=0.4, random_state=42)
    return X_train, X_test, y_train, y_test

def model(X_train, y_train):
    # reg = AdaBoostRegressor(random_state=0, n_estimators=100, loss='linear', learning_rate=0.05 )
    reg = GradientBoostingRegressor(n_estimators=20, learning_rate=0.05, max_depth=2, random_state=0)
    reg = reg.fit(X_train, y_train)
    return reg

# https://www.kaggle.com/davidgasquez/ndcg-scorer
def dcg_score(y_true, y_score, k=5):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)

def test_model(reg, X_test,y_test):
    ids = list(set(X_test["srch_id"]))
    bar = Bar("Processing",max=len(ids))
    scores = []
    for i in ids:
        bar.next()
        X_test["value"] = y_test
        X_test1 = X_test.loc[X_test["srch_id"]==i]
        # print(X_test)
        X = X_test1.drop(["value"],axis=1)
        y = np.asarray(X_test1["value"])
        y_pred = np.asarray(list(reg.predict(X)))
        # print(y)
        # print(y_pred)

        actual = dcg_score(y,y_pred)
        best = dcg_score(y,y)
        # print(actual)
        # print(best)
        if best > 0:
            score = float(actual) / float(best)
        else:
            score = 0
        scores.append(score)
    bar.finish()
    
    return np.mean(scores)

df_train = pd.read_csv('Data/prepro_train.csv')
print("Train loaded")
df_test = pd.read_csv('Data/prepro_test.csv')
print("Test loaded")

df_train = filter_train(df_train,filter_type="balanced")
print("Filtered")

df_train = add_values(df_train)
df_train = df_train.drop(["click_bool","booking_bool","gross_bookings_usd"], axis=1)
df_train = df_train.fillna(-1)
print("Train Cleaned")

df_test = df_test.drop(["click_bool","booking_bool","gross_bookings_usd"], axis=1)
df_test = df_test.fillna(-1)
print("Test Cleaned")

# Testen/valideren op alleen training data
# X_train, X_test, y_train, y_test = train_validation(df_train)

# Trainen op alle data
X_train, y_train = df_train.drop(["value"],axis=1), df_train["value"]
print("Splitted")

# sorteren van submission file op ranking?
reg = model(X_train,y_train)
print("Trained")

# Test model on train data
# NDGC_score = test_model(reg, X_test, y_test)
# print("NDCG score = ", NDGC_score)

y_pred = np.asarray(list(reg.predict(df_test)))

df_test["prediction"] = y_pred
print("Predicted")

df_test = df_test.sort_values(['srch_id', 'prediction'], ascending=[True, False])
print("Sorted")

df_test = df_test[["srch_id","prop_id"]]

df_test.to_csv("Data/submission.csv",index=False)
print("Done")