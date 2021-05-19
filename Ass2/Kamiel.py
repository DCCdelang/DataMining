import pandas as pd
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import seaborn as sns
import operator
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import ndcg_score, make_scorer, SCORERS, accuracy_score, classification_report, precision_score,label_ranking_average_precision_score, make_scorer
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier, StackingRegressor, AdaBoostRegressor, GradientBoostingRegressor,GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.model_selection import GridSearchCV 
# from LambdaRankNN import LambdaRankNN
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.impute import KNNImputer,SimpleImputer
import pyltr

def dcg_score(y_true, y_score, k=5):

    order = np.argsort(y_score)[::-1]

    y_true = np.take(y_true, order[:k])

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)

def add_values(df):
    
    df['value'] = df.apply(lambda row: row.click_bool + (row.booking_bool * 4), axis=1)
    return df

def get_models():
	# models = [ ('knn', KNeighborsClassifier()), ('cart',DecisionTreeClassifier()), ('svm', SVC()), ('bayes',GaussianNB())]

    models = [('rf',RandomForestClassifier(n_estimators=50))]
    return models


def train_class_model_value():
    train = pd.read_csv('Data/fifty_fifty_small.csv')
    train = train.fillna(-1)
    
    features = ['random_bool', 'prob_book', 'srch_length_of_stay', 'srch_booking_window', 'historical_price', 'visitor_hist_starrating', 'srch_query_affinity_score', 'visitor_hist_adr_usd', 'prop_brand_bool', 'prop_review_score', 'prop_review_score_avg', 'srch_adults_count', 'prop_location_score2', 'starrating_diff', 'site_id', 'prop_log_historical_price_avg', 'visitor_location_country_id', 'prop_country_id', 'comp8_rate_percent_diff', 'prop_location_score1', 'prop_location_score1_avg', 'prop_location_score2_avg', 'promotion_flag', 'srch_saturday_night_bool', 'prop_log_historical_price']

    # features = list(train.columns)
    # # features.remove('value')
    # features.remove('click_bool')
    # features.remove('gross_bookings_usd')
    # features.remove('booking_bool')
    # features.remove('srch_id')
    # features.remove('position')

    train = add_values(train)
    X = train[features]
   
    
    y = train["value"]  
    
    # estimators = get_models()

    # clf = StackingClassifier(
    # estimators=estimators,
    # final_estimator=GradientBoostingClassifier(random_state=0, n_estimators=10, learning_rate=0.05 ))

    clf = GradientBoostingClassifier(random_state=0, n_estimators=10, learning_rate=0.05 )
    # clf =  AdaBoostClassifier(random_state=0, n_estimators=20, learning_rate=1 )
    clf = clf.fit(X, y)

    print("Training is done!")
    return clf

# def train_class_model_book():
#     train = pd.read_csv('Data/fifty_fifty_small.csv')
#     train = train.fillna(-1)
#     features = ['random_bool', 'prob_book', 'srch_length_of_stay', 'srch_booking_window', 'historical_price', 'visitor_hist_starrating', 'srch_query_affinity_score', 'visitor_hist_adr_usd', 'prop_brand_bool', 'prop_review_score', 'prop_review_score_avg', 'srch_adults_count', 'prop_location_score2', 'starrating_diff', 'site_id', 'prop_log_historical_price_avg', 'visitor_location_country_id', 'prop_country_id', 'comp8_rate_percent_diff', 'prop_location_score1', 'prop_location_score1_avg', 'prop_location_score2_avg', 'promotion_flag', 'srch_saturday_night_bool', 'prop_log_historical_price']
#     X = train[features]
#     y = train["booking_bool"]  
    
#     clf = GradientBoostingClassifier(random_state=0, n_estimators=5, learning_rate=0.1 )
#     # clf =  AdaBoostClassifier(random_state=0, n_estimators=100, learning_rate=0.05 )

#     clf = clf.fit(X, y)



#     print("Training is done!")
#     return clf


def predict_value(clf):
    test = pd.read_csv('Data/validation_test.csv')

    test = test.fillna(-1)
    features = ['random_bool', 'prob_book', 'srch_length_of_stay', 'srch_booking_window', 'historical_price', 'visitor_hist_starrating', 'srch_query_affinity_score', 'visitor_hist_adr_usd', 'prop_brand_bool', 'prop_review_score', 'prop_review_score_avg', 'srch_adults_count', 'prop_location_score2', 'starrating_diff', 'site_id', 'prop_log_historical_price_avg', 'visitor_location_country_id', 'prop_country_id', 'comp8_rate_percent_diff', 'prop_location_score1', 'prop_location_score1_avg', 'prop_location_score2_avg', 'promotion_flag', 'srch_saturday_night_bool', 'prop_log_historical_price']

    # features = list(test.columns)
    # features.remove('value')
    # features.remove('click_bool')
    # features.remove('gross_bookings_usd')
    # features.remove('booking_bool')
    # features.remove('srch_id')
    # features.remove('position')


    X = test[features]
    y = test['value']
    new_frame = pd.DataFrame()
    predictions = clf.predict(X)

    print(classification_report(y, predictions))
    print("Accuracy = ",accuracy_score(y, predictions))
    
    new_frame['prop_id'] = test['prop_id']
    new_frame['value'] = predictions
    new_frame['srch_id'] = test['srch_id']
    # new_frame = new_frame.sort_values(['srch_id'])
    

    return new_frame

def predict_book(clf, new_frame):
    test = pd.read_csv('Data/validation_test.csv')
    test = test.fillna(-1)
    features = ['random_bool', 'prob_book', 'srch_length_of_stay', 'srch_booking_window', 'historical_price', 'visitor_hist_starrating', 'srch_query_affinity_score', 'visitor_hist_adr_usd', 'prop_brand_bool', 'prop_review_score', 'prop_review_score_avg', 'srch_adults_count', 'prop_location_score2', 'starrating_diff', 'site_id', 'prop_log_historical_price_avg', 'visitor_location_country_id', 'prop_country_id', 'comp8_rate_percent_diff', 'prop_location_score1', 'prop_location_score1_avg', 'prop_location_score2_avg', 'promotion_flag', 'srch_saturday_night_bool', 'prop_log_historical_price']
    X = test[features]
    y = test["booking_bool"] 

    predictions = clf.predict(X)

    print(classification_report(y, predictions))
    print("Accuracy = ",accuracy_score(y, predictions))
    new_frame['booking_bool'] = predictions    

    new_frame['prop_id'] = test['prop_id']
    
    new_frame = new_frame.sort_values(['srch_id'])
    return new_frame

def test_clf_model(new_frame):
    test = pd.read_csv('Data/validation_test.csv')
    
    test = test.fillna(-1)
    scores = []
    p_scores = []
    test = test.sort_values(['srch_id'])

    ids = list(set(test['srch_id']))

    for i in ids:
        test1 = test.loc[test['srch_id'] == i]
        predict1 = new_frame.loc[new_frame['srch_id']==i]
   
       
        true = test1["value"]  
        predict = predict1["value"] 
        
        predict = np.asarray(list(predict))
        true = np.asarray(list(true))

        # print(predict, true)
        # print(predict, true)

        high_score = dcg_score(true, true)
        real_score = dcg_score(true, predict)
        # print(score, high_score)

        if real_score > 0:
            score = float(real_score) / float(high_score)
        else:
            score = 0

        predict = np.asarray([list(predict)])
        true = np.asarray([list(true)])
        p_score = ndcg_score(true, predict, k=10)
    
        p_scores.append(p_score)
        scores.append(score)
    

    print(np.mean(scores))
    print(np.mean(p_scores))


def train_reg_model():
    train = pd.read_csv('Data/fifty_fifty_small3.csv')
    train = train.fillna(-1)
    scaler = MinMaxScaler()
    # features = ['random_bool', 'prob_book', 'srch_length_of_stay', 'srch_booking_window', 'historical_price', 'visitor_hist_starrating', 'srch_query_affinity_score', 'visitor_hist_adr_usd', 'prop_brand_bool', 'prop_review_score', 'prop_review_score_avg', 'srch_adults_count', 'prop_location_score2', 'starrating_diff', 'site_id', 'prop_log_historical_price_avg', 'visitor_location_country_id', 'prop_country_id', 'comp8_rate_percent_diff', 'prop_location_score1', 'prop_location_score1_avg', 'prop_location_score2_avg', 'promotion_flag', 'srch_saturday_night_bool', 'prop_log_historical_price']
    # features = ['random_bool', 'position_mean_extend', 'prob_book', 'prob_click', 'prop_location_score2', 'srch_length_of_stay', 'srch_booking_window', 'prop_location_score2_avg_prop', 'promotion_flag', 'prop_starrating', 'prop_starrating_avg_prop', 'prop_starrating_avg_dest', 'prop_location_score1_avg_dest', 'prop_review_score', 'prop_review_score_avg_prop', 'historical_price', 'visitor_hist_starrating', 'srch_query_affinity_score', 'prop_review_score_avg_srch', 'visitor_hist_adr_usd', 'prop_brand_bool', 'prop_log_historical_price_avg_dest', 'srch_adults_count', 'prop_review_score_avg_dest', 'prop_log_historical_price_avg_prop']
    features = ['random_bool', 'position_mean_extend', 'prob_book', 'prob_click', 'prop_location_score2', 'srch_length_of_stay', 'srch_booking_window', 'prop_location_score2_avg_prop', 'promotion_flag', 'prop_location_score2_median_prop', 'prop_starrating', 'prop_starrating_avg_prop', 'prop_starrating_median_prop', 'prop_location_score1_avg_dest', 'prop_starrating_avg_dest', 'prop_review_score', 'prop_review_score_avg_prop', 'prop_review_score_median_prop', 'prop_location_score2_std_prop', 'historical_price', 'visitor_hist_starrating', 'srch_query_affinity_score', 'prop_review_score_avg_srch', 'visitor_hist_adr_usd', 'prop_brand_bool']
    features = list(train.columns)
    features.remove('click_bool')
    features.remove('gross_bookings_usd')
    features.remove('booking_bool')
    features.remove('srch_id')
    features.remove('position')


    print(len(features))
    train = add_values(train)
    X = train[features]
    # imp = SimpleImputer(strategy="median")
    # X = SimpleImputer(strategy="mean")
    # X = imp.fit_transform(X)
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
    # reg = AdaBoostRegressor(random_state=0, n_estimators=100, loss='linear', learning_rate=0.05 )
    
    # reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 4, alpha = 10, n_estimators = 150, child_weight = 10, gamma = 1, subsample=0.8)
    # # reg  = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 4, alpha = 10, n_estimators = 150)
    # reg = reg.fit(X, y)
    


    reg = xgb.XGBRanker(  
    booster='gbtree',
    objective='rank:ndcg',
    random_state=42, 
    learning_rate=0.1,
    colsample_bytree=0.9, 
    eta=0.05, 
    max_depth=6, 
    n_estimators=110, 
    subsample=0.75 
    )
    groups = train.groupby('srch_id').size().to_frame('size')['size'].to_numpy()
    reg.fit(X, y, group=groups)
    # metric = pyltr.metrics.NDCG(k=5)

    # model = pyltr.models.LambdaMART(
    #             metric=metric,
    #             n_estimators=1000,
    #             learning_rate=0.02,
    #             max_features=0.5,
    #             query_subsample=0.5,
    #             max_leaf_nodes=10,
    #             min_samples_leaf=64,
    #             verbose=1,
    #         )

    # model.fit(X, y, qid)
    # reg = pyltr.models.LambdaMART(random_state=0)
    # reg.fit(X,y,qid)
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

def grid_search():
    '''
    best = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 4, alpha = 10, n_estimators = 150, child_weight = 10, gamma = 1, subsample=0.8)
    '''
    train = pd.read_csv('Data/fifty_fifty_small.csv')
    train = train.fillna(-1)

    # features = ['random_bool', 'prob_book', 'srch_length_of_stay', 'srch_booking_window', 'historical_price', 'visitor_hist_starrating', 'srch_query_affinity_score', 'visitor_hist_adr_usd', 'prop_brand_bool', 'prop_review_score', 'prop_review_score_avg', 'srch_adults_count', 'prop_location_score2', 'starrating_diff', 'site_id', 'prop_log_historical_price_avg', 'visitor_location_country_id', 'prop_country_id', 'comp8_rate_percent_diff', 'prop_location_score1', 'prop_location_score1_avg', 'prop_location_score2_avg', 'promotion_flag', 'srch_saturday_night_bool', 'prop_log_historical_price']
    features = ['random_bool', 'position_mean_extend', 'prob_book', 'prob_click', 'prop_location_score2', 'srch_length_of_stay', 'srch_booking_window', 'prop_location_score2_avg_prop', 'promotion_flag', 'prop_starrating', 'prop_starrating_avg_prop', 'prop_starrating_avg_dest', 'prop_location_score1_avg_dest', 'prop_review_score', 'prop_review_score_avg_prop', 'historical_price', 'visitor_hist_starrating', 'srch_query_affinity_score', 'prop_review_score_avg_srch', 'visitor_hist_adr_usd', 'prop_brand_bool', 'prop_log_historical_price_avg_dest', 'srch_adults_count', 'prop_review_score_avg_dest', 'prop_log_historical_price_avg_prop']
    features = ['random_bool', 'position_mean_extend', 'prob_book', 'prob_click', 'prop_location_score2', 'srch_length_of_stay', 'srch_booking_window', 'prop_location_score2_avg_prop', 'promotion_flag', 'prop_location_score2_median_prop', 'prop_starrating', 'prop_starrating_avg_prop', 'prop_starrating_median_prop', 'prop_location_score1_avg_dest', 'prop_starrating_avg_dest', 'prop_review_score', 'prop_review_score_avg_prop', 'prop_review_score_median_prop', 'prop_location_score2_std_prop', 'historical_price', 'visitor_hist_starrating', 'srch_query_affinity_score', 'prop_review_score_avg_srch', 'visitor_hist_adr_usd', 'prop_brand_bool']

    # features = list(train.columns)
    # features.remove('click_bool')
    # features.remove('gross_bookings_usd')
    # features.remove('booking_bool')
    # features.remove('srch_id')
    # features.remove('position')
    train = add_values(train)
    X = train[features]
    y = train["value"]  
    learning_rate = [0.05, 0.1, 0.2]
    child_weight = [1, 5, 10]
    gamma =  [0, 1, 5]
    subsample = [0.8, 1.0]
    colsample_bytree= [0.3, 0.6, 0.8]
    max_depth = [3, 4, 5]
    n_estimators = [50, 100, 150]
    lists = []

    for d in colsample_bytree:
        for c in subsample:
            for b in gamma:  
                for a in child_weight:
                    lists.append([a, b, c, d])
    random.shuffle(lists)

    param_scores = {}
    for params in lists:
        print(params)
        reg  = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = params[-1], learning_rate = 0.1, max_depth = 4, alpha = 10, n_estimators = 150, gamma = params[1], child_weight=params[0], subsample = params[2])
        reg = reg.fit(X, y)
        score = test_reg_model(reg)
        print('score: ', score)
        param_scores[str(params)] = score


    print(max(param_scores.iteritems(), key=operator.itemgetter(1))[0])

def test_reg_model(reg):
    
    test = pd.read_csv('Data/validation_test3.csv')
    
    # test = test.fillna(-1)
    
    p_scores = []
    scores = []
    # features = ['random_bool', 'position_mean_extend', 'prob_book', 'prob_click', 'prop_location_score2', 'srch_length_of_stay', 'srch_booking_window', 'prop_location_score2_avg_prop', 'promotion_flag', 'prop_location_score2_median_prop', 'prop_starrating', 'prop_starrating_avg_prop', 'prop_starrating_median_prop', 'prop_location_score1_avg_dest', 'prop_starrating_avg_dest', 'prop_review_score', 'prop_review_score_avg_prop', 'prop_review_score_median_prop', 'prop_location_score2_std_prop', 'historical_price', 'visitor_hist_starrating', 'srch_query_affinity_score', 'prop_review_score_avg_srch', 'visitor_hist_adr_usd', 'prop_brand_bool']
    # features = ['random_bool', 'prob_book', 'srch_length_of_stay', 'srch_booking_window', 'historical_price', 'visitor_hist_starrating', 'srch_query_affinity_score', 'visitor_hist_adr_usd', 'prop_brand_bool', 'prop_review_score', 'prop_review_score_avg', 'srch_adults_count', 'prop_location_score2', 'starrating_diff', 'site_id', 'prop_log_historical_price_avg', 'visitor_location_country_id', 'prop_country_id', 'comp8_rate_percent_diff', 'prop_location_score1', 'prop_location_score1_avg', 'prop_location_score2_avg', 'promotion_flag', 'srch_saturday_night_bool', 'prop_log_historical_price']
    # features = ['random_bool', 'position_mean_extend', 'prob_book', 'prob_click', 'prop_location_score2', 'srch_length_of_stay', 'srch_booking_window', 'prop_location_score2_avg_prop', 'promotion_flag', 'prop_starrating', 'prop_starrating_avg_prop', 'prop_starrating_avg_dest', 'prop_location_score1_avg_dest', 'prop_review_score', 'prop_review_score_avg_prop', 'historical_price', 'visitor_hist_starrating', 'srch_query_affinity_score', 'prop_review_score_avg_srch', 'visitor_hist_adr_usd', 'prop_brand_bool', 'prop_log_historical_price_avg_dest', 'srch_adults_count', 'prop_review_score_avg_dest', 'prop_log_historical_price_avg_prop']
    features = list(test.columns)
    features.remove('click_bool')
    features.remove('gross_bookings_usd')
    features.remove('booking_bool')
    features.remove('srch_id')
    features.remove('position')
    features.remove('value')


    # imputer = SimpleImputer(strategy="median")
    # X = SimpleImputer(strategy="mean")
  
    print(len(features))

    # def predict(model, df):
    #     return model.predict(df.loc[:, ~df.columns.isin(['srch_id'])])
  
    # predictions = (test.groupby('srch_id')
    #            .apply(lambda test: predict(reg, test)))
    # test.loc[:, test.columns != 'value'] = imputer.fit_transform(test.loc[:, test.columns != 'value'])
    ids = list(set(test['srch_id']))
    for i in ids:
        test1 = test.loc[test['srch_id'] == i]

   
        X = test1[features]
        
        y = test1["value"]  
        # y_pred = reg.predict(X)
        
        predictions = reg.predict(X)
        # print(predictions)
        true = np.asarray([y])
        # predict = np.asarray([list(predictions)])


        # score = ndcg_score(true, predict, k=5)

        
        # predict = np.asarray([list(predictions)])
        # true = np.asarray([list(true)])
        predict = np.asarray(list(predictions))
        true = np.asarray(list(true))

        # print(predict ,true)

        # print(predict, true)
        # print(predict, true)

        high_score = dcg_score(true, true, k=5)
        real_score = dcg_score(true, predict, k=5)
        # print(score, high_score)

        if real_score > 0:
            score = float(real_score) / float(high_score)
        else:
            score = 0
        # print(true, predict)
        # score = ndcg_score(true, predict, k=4)
        scores.append(score)
    # print('hallo')
    # print(np.mean(scores))
    return np.mean(scores)

    

def make_submission_file():
    train = pd.read_csv('Data/fifty_fifty3.csv')
    # train = train.fillna(-1)
    train = add_values(train)
    features = list(train.columns)
    features.remove('value')
    features.remove('click_bool')
    features.remove('gross_bookings_usd')
    features.remove('booking_bool')
    features.remove('srch_id')
    features.remove('position')
    # features = ['random_bool', 'prob_book', 'srch_length_of_stay', 'srch_booking_window', 'historical_price', 'visitor_hist_starrating', 'srch_query_affinity_score', 'visitor_hist_adr_usd', 'prop_brand_bool', 'prop_review_score', 'prop_review_score_avg', 'srch_adults_count', 'prop_location_score2', 'starrating_diff', 'site_id', 'prop_log_historical_price_avg', 'visitor_location_country_id', 'prop_country_id', 'comp8_rate_percent_diff', 'prop_location_score1', 'prop_location_score1_avg', 'prop_location_score2_avg', 'promotion_flag', 'srch_saturday_night_bool', 'prop_log_historical_price']
    # features = ['random_bool', 'position_mean_extend', 'prob_book', 'prob_click', 'prop_location_score2', 'srch_length_of_stay', 'srch_booking_window', 'prop_location_score2_avg_prop', 'promotion_flag', 'prop_starrating', 'prop_starrating_avg_prop', 'prop_starrating_avg_dest', 'prop_location_score1_avg_dest', 'prop_review_score', 'prop_review_score_avg_prop', 'historical_price', 'visitor_hist_starrating', 'srch_query_affinity_score', 'prop_review_score_avg_srch', 'visitor_hist_adr_usd', 'prop_brand_bool', 'prop_log_historical_price_avg_dest', 'srch_adults_count', 'prop_review_score_avg_dest', 'prop_log_historical_price_avg_prop']
    # features.remove('Unnamed: 0')
    
    X = train[features]
    y = train["value"]

    # estimators = [
    # ('rf', RandomForestRegressor(n_estimators = 50, random_state=0))]
   
    # reg = StackingRegressor(
    # estimators=estimators,
    # final_estimator=GradientBoostingRegressor(n_estimators=50, learning_rate=1.0, max_depth=2, random_state=0))

    # reg  = LambdaRankNN(input_size=X.shape[1], hidden_layer_sizes=(16,8,), activation=('relu', 'relu',), solver='adam')
    # qid = np.asarray(train['srch_id'], dtype = np.int64)

    # reg.fit(np.asarray(X, dtype = np.int64), np.asarray(y, dtype = np.int64), qid, epochs=5)
    
    # reg  = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 100)
    reg = xgb.XGBRanker(  
    booster='gbtree',
    objective='rank:ndcg',
    random_state=42, 
    learning_rate=0.1,
    colsample_bytree=0.9, 
    eta=0.05, 
    max_depth=6, 
    n_estimators=110, 
    subsample=0.75 
    )
    groups = train.groupby('srch_id').size().to_frame('size')['size'].to_numpy()
    reg = reg.fit(X, y, group=groups)
    # reg = reg.fit(X, y)
    
    test = pd.read_csv('Data/prepro_test3.csv')
    test = test.fillna(-1)
    df_sub = test
    
    X = test[features] 
    predictions = reg.predict(X)

    df_sub['predicted_values'] = predictions

    df_sub = df_sub.sort_values(['srch_id', 'predicted_values'], ascending=[True, False])
    
    df_sub = df_sub[['srch_id', 'prop_id']]

    df_sub["srch_id"] = pd.to_numeric(df_sub["srch_id"],downcast='integer')

    df_sub.to_csv('Data/submission_XGBOOSTRANK.csv', index=False)


    # print(df.columns,df1.columns)


if __name__ == "__main__":

    # print(test_reg_model(train_reg_model()))
    make_submission_file()

    # df = pd.read_csv('Data/submission_GXBOOST_new_features.csv')
    # first_column = df.columns[0]
    # # Delete first
    # df = df.drop([first_column], axis=1)
    # df.to_csv('Data/submission_GXBOOST_new_features.csv', index=False)

    # print(df.head(5))
   
    # print(df.head(5))
    # df.to_csv('Data/submission_GXBOOST_new_features.csv')
    # new_frame = predict_value(train_class_model_value())
    # # new_frame = predict_book(train_class_model_book(), new_frame)

    # test_clf_model(new_frame)
    # print(new_frame.head(5))

    # new_frame = new_frame.sort_values(['srch_id', 'value'], ascending=[True, False])

    # new_frame = new_frame[['srch_id', 'prop_id']]
    # new_frame.to_csv('sumbission_test.csv', index=False)
    # grid_search()