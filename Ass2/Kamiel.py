import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.metrics import ndcg_score, make_scorer, SCORERS
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import GridSearchCV 

def train_model():
    train = pd.read_csv('Data/validation_train.csv')
    train = train.fillna(0)

    features = list(train.columns)
    features.remove('value')
    features.remove('click_bool')
    features.remove('gross_bookings_usd')
    features.remove('booking_bool')
    features.remove('Unnamed: 0')
    features.remove('Unnamed: 0.1')
    features.remove('Unnamed: 0.1.1')
    features.remove('ra')
    features.remove('srch_id')

    X = train[features]
    y = train["value"]  
    estimators = [
    ('rf', RandomForestRegressor(n_estimators = 50, random_state=0))]
   
    reg = StackingRegressor(
    estimators=estimators,
    final_estimator=GradientBoostingRegressor(n_estimators=50, learning_rate=1.0, max_depth=2, random_state=0))
    
    # reg = GradientBoostingRegressor(n_estimators=50, learning_rate=1.0, max_depth=2, random_state=0)
    reg = reg.fit(X, y)

    
    # hyperparameters = {                     
    #                 'n_estimators': [50,100,150],
    #                 'max_depth': [2, 4],
    #                 'learning_rate': [1, 2],
    #                 'criterion': ['friedman_mse', 'mse', 'mae']
    #             }

    # print(SCORERS.keys())
    # reg = GridSearchCV(reg, hyperparameters, scoring='neg_mean_squared_error')
    # # Fit and tune model
    # print(reg.get_params().keys())
    # reg.fit(X, y)


    # print(reg.best_params_)

    print("Training is done!")
    return reg

def test_model(reg):
    
    test = pd.read_csv('Data/validation_test.csv')
    
    test = test.fillna(0)
    scores = []

    ids = list(set(test['ra']))
    features = list(test.columns)
    features.remove('value')
    features.remove('click_bool')
    features.remove('gross_bookings_usd')
    features.remove('booking_bool')
    features.remove('Unnamed: 0')
    features.remove('Unnamed: 0.1')
    features.remove('Unnamed: 0.1.1')
    features.remove('ra')
    features.remove('srch_id')
    print(features)

    for i in ids:
        test1 = test.loc[test['ra'] == i]

   
        X = test1[features]
        y = test1["value"]  
        
        predictions = reg.predict(X)

        true = np.asarray([y])
        predict = np.asarray([list(predictions)])
        score = ndcg_score(true, predict)
        scores.append(score)
    print('hallo')
    print(np.mean(scores))

def make_submission_file():
    train = pd.read_csv('Data/clicked_data.csv')
    train = train.fillna(0)
    features = list(train.columns)
    features.remove('value')
    features.remove('click_bool')
    features.remove('gross_bookings_usd')
    features.remove('booking_bool')
    features.remove('Unnamed: 0')
    features.remove('ra')
    features.remove('srch_id')
    
    X = train[features]
    y = train["value"]

    estimators = [
    ('rf', RandomForestRegressor(n_estimators = 50, random_state=0))]
   
    reg = StackingRegressor(
    estimators=estimators,
    final_estimator=GradientBoostingRegressor(n_estimators=50, learning_rate=1.0, max_depth=2, random_state=0))
    reg = reg.fit(X, y)
    
    test = pd.read_csv('Data/prepro_test.csv')
    test = test.fillna(0)
    df_sub = test
    
    X = test[features] 
    predictions = reg.predict(X)

    df_sub['predicted_values'] = predictions

    df_sub = df_sub.sort_values(['srch_id', 'predicted_values'], ascending=[True, False])
    
    df_sub = df_sub[['srch_id', 'prop_id']]

    df_sub.to_csv('Data/submission1.csv', index=False)


    # print(df.columns,df1.columns)

test_model(train_model())