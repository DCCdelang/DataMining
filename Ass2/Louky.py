import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.metrics import ndcg_score, make_scorer, SCORERS
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import GridSearchCV 

import pyltr

def train_model():
    train = pd.read_csv('Data/validation_train.csv')
    train = train.fillna(-2)

    features = list(train.columns)
    features.remove('value')
    features.remove('click_bool')
    features.remove('gross_bookings_usd')
    features.remove('booking_bool')
    features.remove('srch_id')
    features.remove('position')

    X = train[features]
    y = train["value"]  
    estimators = [
    ('rf', RandomForestRegressor(n_estimators = 200, random_state=0))]
   
    reg = StackingRegressor(
    estimators=estimators,
    final_estimator=GradientBoostingRegressor(n_estimators=10, learning_rate=1.0, max_depth=2, random_state=0))
    
    # reg = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=2, random_state=0)
    reg = reg.fit(X, y)

    
    # hyperparameters = {                     
    #                 'n_estimators': [10,20,30,40,50],
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
    
    test = test.fillna(-2)
    scores = []

    ids = list(set(test['srch_id']))
    features = list(test.columns)
    features.remove('value')
    features.remove('click_bool')
    features.remove('gross_bookings_usd')
    features.remove('booking_bool')
    features.remove('srch_id')
    features.remove('position')
    print(features)

    for i in ids:
        test1 = test.loc[test['srch_id'] == i]

   
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
    train = pd.read_csv('Data/clicked_data_submission.csv', sep=',')
    train = train.fillna(0)
    features = list(train.columns)
    features.remove('value')
    features.remove('click_bool')
    features.remove('gross_bookings_usd')
    features.remove('booking_bool')
    features.remove('srch_id')
    features.remove('position')
    
    
    #features.remove('site_id')
    #features.remove('viVsitor_location_country_id')
    #features.remove('visitor_hist_starrating')
    
    #features.remove('visitor_hist_adr_usd')
    #features.remove('prop_country_id')
    #features.remove('prop_id')
    #features.remove('prop_starrating')
    #features.remove('prop_review_score')
    
    Tqids = np.array(train['srch_id'])
    
    X = np.array(train[features])
    y = np.array(train["value"])
    
    #TX, Ty, Tqids, _ = pyltr.data.letor.read_dataset(X)
    
    metric = pyltr.metrics.NDCG()

    # Only needed if you want to perform validation (early stopping & trimming)
    # monitor = pyltr.models.monitors.ValidationMonitor(
    #    X, y, Vqids, metric=metric, stop_after=250)

    model = pyltr.models.LambdaMART(
        metric=metric,
        n_estimators=100,
        learning_rate=0.02,
        max_features=0.5,
        query_subsample=0.5,
        max_leaf_nodes=10,
        min_samples_leaf=64,
        verbose=1,
        )

    model.fit(X, y, Tqids)
    
    
    test = pd.read_csv('Data/prepro_test.csv')
    test = test.fillna(-2)
    df_sub = test
    
    X_p = test[features]
    ids = test['srch_id']
    predictions = model.predict(X_p)
    df_sub['predicted_values'] = predictions
    df_sub2 = df_sub.sort_values(['srch_id', 'predicted_values'], ascending=[True, False])
    df_sub2 = df_sub2[['srch_id', 'prop_id']]
    df_sub2.to_csv('Data/submission_lambdamart1.csv', index=False)
    
    
    # estimators = [
    # ('rf', RandomForestRegressor(n_estimators = 50, random_state=0))]
   
    # reg = StackingRegressor(
    # estimators=estimators,
    # final_estimator=GradientBoostingRegressor(n_estimators=50, learning_rate=1.0, max_depth=2, random_state=0))

    """
    reg = GradientBoostingRegressor(n_estimators=5, learning_rate=1.0, max_depth=2, random_state=0)
    reg = reg.fit(X, y)
    
    test = pd.read_csv('Data/prepro_test.csv')
    test = test.fillna(-2)
    df_sub = test
    
    X = test[features] 
    predictions = reg.predict(X)

    df_sub['predicted_values'] = predictions

    df_sub = df_sub.sort_values(['srch_id', 'predicted_values'], ascending=[True, False])
    
    df_sub = df_sub[['srch_id', 'prop_id']]

    df_sub.to_csv('Data/submission1.csv', index=False)


    # print(df.columns,df1.columns)
    """

make_submission_file()


    

