import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.metrics import ndcg_score, make_scorer, SCORERS
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import GridSearchCV 

def make_files():
    df = pd.read_csv('Data/training_set_VU_DM.csv')

    df = df.tail(100000)
    df.to_csv('Data/train_head.csv', index=False)

def add_values(df, name):
    values = []
    for i in range(len(df['click_bool'])):
        value = 0
        if df.iloc[i]['click_bool'] != 0:
            value +=1
        if df.iloc[i]['booking_bool'] != 0:
            value +=4
        values.append(value)

    df['value'] = values
    df.to_csv(name, index=False)
       
def train_model():
    train = pd.read_csv('Data/train_data.csv')
    train = train.fillna(0)

    features = list(train.columns)
  
    features.remove('position')
    features.remove('click_bool')
    features.remove('gross_bookings_usd')
    features.remove('value')
    features.remove('date_time')
    features.remove('date')
    features.remove('time')
    features.remove('hour')
    features.remove('log_hist_price_dif')
    features.remove('log_price_usd')
    features.remove('price_per_day')
    features.remove('booking_bool')
    features.remove('gross_bookings_per_day')
    # features.remove('prob_book')
    # features.remove('value')
    
    estimators = [
    ('rf', RandomForestRegressor(n_estimators = 50, random_state=0))]
    X = train[features]
    y = train["value"]  
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
    
    test = pd.read_csv('Data/test_data.csv')
    
    test = test.fillna(0)
    scores = []
    ids = list(set(test['ra']))
    features = list(test.columns)
    
    features.remove('position')
    features.remove('click_bool')
    features.remove('booking_bool')
    features.remove('gross_bookings_usd')
    features.remove('value')
    features.remove('date_time')
    features.remove('date')
    features.remove('time')
    features.remove('hour')
    features.remove('log_hist_price_dif')
    features.remove('log_price_usd')
    features.remove('gross_bookings_per_day')
    # features.remove('prob_book')

    

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

def feature_importance():

    train = pd.read_csv('Data/train_data.csv')
    train = train.fillna(0)
    feat = list(train.columns)

    feat.remove('position')
    feat.remove('click_bool')
    feat.remove('gross_bookings_usd')
    feat.remove('value')
    feat.remove('date_time')
    feat.remove('date')
    feat.remove('time')
    feat.remove('hour')
    feat.remove('log_hist_price_dif')
    

  
    features = feat[1:-1]

    X = train[features]
    y = train["value"]  

    reg = GradientBoostingRegressor(n_estimators=1000, learning_rate=1.0,
    max_depth=1, random_state=0)

    reg = reg.fit(X, y)
    feature_importance = reg.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, np.array(feat)[sorted_idx])
    plt.title('Feature Importance (MDI)')

    result = permutation_importance(reg, X, y, n_repeats=10,
                                    random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()
    plt.subplot(1, 2, 2)
    plt.boxplot(result.importances[sorted_idx].T,
                vert=False, labels=np.array(feat)[sorted_idx])
    plt.title("Permutation Importance (test set)")
    fig.tight_layout()
    plt.show()

def make_submission_file():
    train = pd.read_csv('Data/clicked_data.csv')
    train = train.fillna(0)
    features = list(train.columns)

    features.remove('position')
    features.remove('click_bool')
    
    features.remove('gross_bookings_usd')
    features.remove('value')
    features.remove('date_time')
    features.remove('date')
    features.remove('time')
    features.remove('hour')
    features.remove('log_hist_price_dif')
    features.remove('log_price_usd')
    features.remove('booking_bool')
    features.remove('gross_bookings_per_day')
    # features = ['price_usd', 'prop_review_score', 'prop_starrating','promotion_flag', 'prop_starrating','srch_saturday_night_bool']
    X = train[features]
    y = train["value"]  
    reg = GradientBoostingRegressor(n_estimators=1000, learning_rate=1.0,
    max_depth=1, random_state=0)
    reg = reg.fit(X, y)
    
    test = pd.read_csv('Data/processed_end_data.csv')
    test = test.fillna(0)
    df_sub = test
    
    X = test[features] 
    predictions = reg.predict(X)

    df_sub['predicted_values'] = predictions

    df_sub = df_sub.sort_values(['srch_id', 'predicted_values'], ascending=[True, False])
    
    df_sub = df_sub[['srch_id', 'prop_id']]

    df_sub.to_csv('Data/submission1.csv', index=False)

def plot(df):
    
    df['price_usd'] = df['price_usd'].clip(0, 1500)
    # df = df.head(5000)
    sns.histplot(data=df, x = 'price_usd')
    plt.show()

    sns.histplot(data=df, x = 'visitor_hist_starrating')
    plt.show()

    sns.histplot(data=df, x ='prop_starrating')
    plt.show()

    sns.histplot(data=df, x = 'promotion_flag')
    plt.show()

def drop_nan_columns():
    df = pd.read_csv('Data/training_set_VU_DM.csv')
    # plot(df)
    print(df.shape)
    df1 = df.dropna(axis=1, thresh= 0.1 * df.shape[0])
    print(df1.shape)
    for i in list(df.columns):
        if i not in (df1.columns):
            print(i)
    df1.to_csv('Data/training_set_VU_DM_deleted.csv', index=False)
    # print(df.columns,df1.columns)
def check_same_values():
    train = pd.read_csv('Data/training_set_VU_DM.csv')
    test = pd.read_csv('Data/test_set_VU_DM.csv')
    prop_train = set(train['prop_id'])
    prop_test = set(test['prop_id'])
    print(len(list(prop_train - prop_test)))

def random():
    df = pd.read_csv('Data/clicked_data.csv')
    print(df['random_bool'].sum())


# feature_importance()


make_files()
# df = pd.read_csv('Data/test_data.csv')
# add_values(df, 'Data/test_data.csv')
# test_model(train_model())

# drop_nan_columns()

# check_same_values()
# make_submission_file()

# train_model()
# random()

