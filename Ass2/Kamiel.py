import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.metrics import ndcg_score, make_scorer, SCORERS
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, AdaBoostRegressor
from sklearn.model_selection import GridSearchCV 
from LambdaRankNN import LambdaRankNN

def train_model():
    train = pd.read_csv('Data/validation_train_clicked.csv')
    train = train.fillna(-1)

    features = list(train.columns)
    features.remove('value')
    features.remove('click_bool')
    features.remove('gross_bookings_usd')
    features.remove('booking_bool')
    # features.remove('srch_id')
    features.remove('position')
    features = ['random_bool', 'prob_book', 'srch_length_of_stay', 'srch_booking_window', 'historical_price', 'visitor_hist_starrating', 'srch_query_affinity_score', 'visitor_hist_adr_usd', 'prop_brand_bool', 'prop_review_score', 'prop_review_score_avg', 'srch_adults_count', 'prop_location_score2', 'starrating_diff', 'site_id', 'prop_log_historical_price_avg', 'visitor_location_country_id', 'prop_country_id', 'comp8_rate_percent_diff', 'prop_location_score1', 'prop_location_score1_avg', 'prop_location_score2_avg', 'promotion_flag', 'srch_saturday_night_bool', 'prop_log_historical_price']
    X = train[features]
    y = train["value"]  
    # estimators = [
    # ('rf', RandomForestRegressor(n_estimators = 100, random_state=0)), 
    # ('gb',GradientBoostingRegressor(n_estimators=10, learning_rate=1.0, max_depth=2, random_state=0))
    # ]
   
    # reg = StackingRegressor(
    # estimators=estimators,
    # final_estimator=AdaBoostRegressor(random_state=0, n_estimators=100, loss='linear'))
    
    # reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=2, random_state=0)
    # reg = RandomForestRegressor(n_estimators = 100, random_state=0)
    # reg = MLPRegressor(random_state=0, max_iter=500)
    # reg  = LambdaRankNN(input_size=X.shape[1], hidden_layer_sizes=(16,8,), activation=('relu', 'relu',), solver='adam')
    # qid = np.asarray(train['srch_id'], dtype = np.int64)

    # reg.fit(np.asarray(X, dtype = np.int64), np.asarray(y, dtype = np.int64), qid, epochs=5)
    reg = AdaBoostRegressor(random_state=0, n_estimators=100, loss='linear', learning_rate=0.05 )
    reg = reg.fit(X, y)

    
    # hyperparameters = {                     
    #                 'n_estimators': [10, 50, 100, 200],
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
    
    test = pd.read_csv('Data/validation_test_small.csv')
    
    test = test.fillna(-1)
    scores = []

    ids = list(set(test['srch_id']))
    features = list(test.columns)
    features.remove('value')
    features.remove('click_bool')
    features.remove('gross_bookings_usd')
    features.remove('booking_bool')
    # features.remove('srch_id')
    features.remove('position')
    features = ['random_bool', 'prob_book', 'srch_length_of_stay', 'srch_booking_window', 'historical_price', 'visitor_hist_starrating', 'srch_query_affinity_score', 'visitor_hist_adr_usd', 'prop_brand_bool', 'prop_review_score', 'prop_review_score_avg', 'srch_adults_count', 'prop_location_score2', 'starrating_diff', 'site_id', 'prop_log_historical_price_avg', 'visitor_location_country_id', 'prop_country_id', 'comp8_rate_percent_diff', 'prop_location_score1', 'prop_location_score1_avg', 'prop_location_score2_avg', 'promotion_flag', 'srch_saturday_night_bool', 'prop_log_historical_price']
    # qid = np.asarray(test['srch_id'], dtype = np.int64)
    # X = np.asarray(test[features], dtype = np.int64)
    # y = np.asarray(test["value"], dtype = np.int64)  
    # reg.evaluate(X, y, qid, eval_at=2)
    for i in ids:
        test1 = test.loc[test['srch_id'] == i]

   
        X = test1[features]
        y = test1["value"]  
        # y_pred = reg.predict(X)
        
        predictions = reg.predict(X)

        true = np.asarray([y])
        predict = np.asarray([list(predictions)])
        score = ndcg_score(true, predict)
        scores.append(score)
    print('hallo')

    print(np.mean(scores))

def make_submission_file():
    train = pd.read_csv('Data/train_submission.csv')
    train = train.fillna(-1)
    features = list(train.columns)
    features.remove('value')
    features.remove('click_bool')
    features.remove('gross_bookings_usd')
    features.remove('booking_bool')
    features.remove('srch_id')
    features.remove('position')
    features = ['random_bool', 'prob_book', 'srch_length_of_stay', 'srch_booking_window', 'historical_price', 'visitor_hist_starrating', 'srch_query_affinity_score', 'visitor_hist_adr_usd', 'prop_brand_bool', 'prop_review_score', 'prop_review_score_avg', 'srch_adults_count', 'prop_location_score2', 'starrating_diff', 'site_id', 'prop_log_historical_price_avg', 'visitor_location_country_id', 'prop_country_id', 'comp8_rate_percent_diff', 'prop_location_score1', 'prop_location_score1_avg', 'prop_location_score2_avg', 'promotion_flag', 'srch_saturday_night_bool', 'prop_log_historical_price']
    # features.remove('Unnamed: 0')
    
    X = train[features]
    y = train["value"]

    # estimators = [
    # ('rf', RandomForestRegressor(n_estimators = 50, random_state=0))]
   
    # reg = StackingRegressor(
    # estimators=estimators,
    # final_estimator=GradientBoostingRegressor(n_estimators=50, learning_rate=1.0, max_depth=2, random_state=0))

    reg  = LambdaRankNN(input_size=X.shape[1], hidden_layer_sizes=(16,8,), activation=('relu', 'relu',), solver='adam')
    qid = np.asarray(train['srch_id'], dtype = np.int64)

    reg.fit(np.asarray(X, dtype = np.int64), np.asarray(y, dtype = np.int64), qid, epochs=5)

    reg = AdaBoostRegressor(random_state=0, n_estimators=100, loss='linear', learning_rate=0.05 )
    reg = reg.fit(X, y)
    
    test = pd.read_csv('Data/prepro_test2.csv')
    test = test.fillna(-1)
    df_sub = test
    
    X = test[features] 
    predictions = reg.predict(np.asarray(X, dtype = np.int64))

    df_sub['predicted_values'] = predictions

    df_sub = df_sub.sort_values(['srch_id', 'predicted_values'], ascending=[True, False])
    
    df_sub = df_sub[['srch_id', 'prop_id']]

    df_sub.to_csv('Data/submission1.csv', index=False)


    # print(df.columns,df1.columns)

test_model(train_model())
# make_submission_file()

