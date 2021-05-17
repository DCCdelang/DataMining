import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from numpy.random import RandomState
from progress.bar import Bar
import xgboost as xgb
from sklearn.metrics import make_scorer
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_rows', None)

rs = RandomState(42)

def add_values(df_train):
    df_train['value'] = df_train.apply(lambda row: row.click_bool + (row.booking_bool * 4), axis=1)
    return df_train

def filter_train(df_train,filter_type): # Choose Balanced, clicked
    if filter_type == "None":
        print("# No filter")
    if filter_type == "balanced":
        df_train1 = df_train.loc[df_train["click_bool"]==1]
        print(df_train1.shape)
        df_train2 = df_train[df_train["srch_id"].isin(df_train1["srch_id"])]
        size = 5        # sample size
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

def model(X_train, y_train, n_estimator=150):
    # reg = AdaBoostRegressor(random_state=0, n_estimators=100, loss='linear', learning_rate=0.05 )
    # reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=2, random_state=42)
    # estimators = [
    # ('rf', RandomForestRegressor(n_estimators = 100, random_state=42)), 
    # ('gb',GradientBoostingRegressor(n_estimators=10, learning_rate=1.0, max_depth=2, random_state=42))
    # ]
   
    # reg = StackingRegressor(
    # estimators=estimators,
    # final_estimator=AdaBoostRegressor(random_state=42, n_estimators=100, loss='linear'))
    
    reg = xgb.XGBRegressor(n_estimators = n_estimator, n_jobs=1, random_state = 42)

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
    X_test["value"] = y_test
    Ids = X_test.groupby("srch_id")
    bar = Bar("Processing",max=len(Ids))
    scores = []
    for _, group in Ids:
        bar.next()
        X_test1 = group
        # print(X_test1)
        X = X_test1.drop(["value"],axis=1)
        y = np.asarray(X_test1["value"])
        y_pred = np.asarray(list(reg.predict(X)))

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

def feature_importance(reg):
    xgb.plot_importance(reg,importance_type ="gain")
    plt.show()
    feature_important = reg.get_booster().get_score(importance_type='weight')
    keys = list(feature_important.keys())
    values = list(feature_important.values())

    data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
    data.plot(kind='barh')
    plt.show()

"""WHAT YOU GONNA DO? SUBMIT OR TEST?"""

I_want = "Test" # "Submit" or "Test"
print("This is for a", I_want)

df_train = pd.read_csv('Data/prepro_train.csv')
print("Train loaded")
# df_test = pd.read_csv('Data/prepro_test.csv')
# print("Test loaded")

df_train = filter_train(df_train,filter_type="None")
print("Filtered")

df_train = add_values(df_train)
# df_train = df_train.drop(["click_bool","booking_bool","gross_bookings_usd"], axis=1)
# df_train = df_train.fillna(-1)
print("Train Cleaned")

print(df_train.isna().sum())

# df_test = df_test.fillna(-1)
# print("Test Cleaned")

if I_want == "Test":
    # Testen/valideren op alleen training data
    X_train, X_test, y_train, y_test = train_validation(df_train)
    X_train["value"] = y_train
    X_train = filter_train(X_train,filter_type="balanced")
    X_train = X_train.drop(["click_bool","booking_bool","gross_bookings_usd","position"], axis=1)
    X_test = X_test.drop(["click_bool","booking_bool","gross_bookings_usd","position"], axis=1)
    y_train = X_train["value"]
    X_train = X_train.drop(["value"],axis=1)
    print("Splitted")

if I_want == "Submit":
    # Trainen op alle data
    X_train, y_train = df_train.drop(["value"],axis=1), df_train["value"]
    print("Splitted")

# sorteren van submission file op ranking?
reg = model(X_train,y_train)
print("Trained")

if I_want == "Test":
    # B = ["gbtree", "gblinear", "dart"]
    # N_est = [50, 100, 150]
    # for booster in B:
    #     for n_est in N_est:
    #         print("Param: B=",booster,"n_est=",n_est)
    #         reg = model(X_train,y_train,n_est,booster)
    #         print("Trained")
    # Test model on train data
    NDGC_score = test_model(reg, X_test, y_test)
    print("NDCG score = ", NDGC_score)

if I_want == "Submit":
    y_pred = np.asarray(list(reg.predict(df_test)))

    df_test["prediction"] = y_pred
    print("Predicted")

    df_test = df_test.sort_values(['srch_id', 'prediction'], ascending=[True, False])
    print("Sorted")

    df_test["srch_id"] = pd.to_numeric(df_test["srch_id"],downcast='integer')

    df_test1 = df_test[["srch_id","prop_id"]]

    df_test1.to_csv("Data/submission3.csv",index=False)
    print("Almost done")

    df_test2 = df_test[["srch_id","prop_id","prediction"]]

    df_test2.to_csv("Data/submission3_predictions.csv",index=False)
    print("Done")

feature_importance(reg)
print("plotted")