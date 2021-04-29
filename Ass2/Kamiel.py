import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GridSearchCV 

def make_files():

    df = pd.read_csv('Data/clicked_data.csv')

    df = df.tail(100000)
    df.to_csv('Data/train_data.csv', index=False)

def add_values(df, name):

    values = []
    for i in range(len(df['ra'])):
        value = 0
        if df.iloc[i]['click_bool'] != 0:
            value +=1
        if df.iloc[i]['booking_bool'] != 0:
            value +=4
        values.append(value)

    df['value'] = values
    df.to_csv('Data/test_data.csv', index=False)
       
def train_model():
    train = pd.read_csv('Data/train_data.csv')
    train = train.fillna(0)

    features = ['price_usd', 'prop_review_score', 'prop_starrating','promotion_flag', 'prop_starrating','srch_saturday_night_bool','srch_room_count','srch_children_count']
    X = train[features]
    y = train["value"]  
    reg = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    reg = reg.fit(X, y)
    hyperparameters = {                     
                    'classifier__n_estimators': [50,100,150],
                    'classifier__max_depth': [2, 4],
                    'classifier__min_samples_leaf': [2, 4],
                    'classifier__criterion': ['gini', 'entropy'],
                }

    reg = GridSearchCV(reg, hyperparameters, )
    # Fit and tune model
    clf.fit(X_train, y_train)


    print(clf.best_params_)
    return reg

def test_model(reg):
    
    test = pd.read_csv('Data/test_data.csv')
    
    test = test.fillna(0)
    scores = []
    ids = list(set(test['ra']))

    for i in ids:
        test1 = test.loc[test['ra'] == i]

        features = ['price_usd', 'prop_review_score', 'prop_starrating','promotion_flag', 'prop_starrating','srch_saturday_night_bool','srch_room_count','srch_children_count']
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
    feat.remove('booking_bool')
    feat.remove('srch_saturday_night_bool')
    feat.remove('srch_room_count')
    feat.remove('srch_children_count')
        
  
    features = feat[3:-1]
    X = train[features]
    y = train["value"]  

    reg = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0,
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

    features = ['price_usd', 'prop_review_score', 'prop_starrating','promotion_flag', 'prop_starrating','srch_saturday_night_bool']
    X = train[features]
    y = train["value"]  
    reg = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0,
    max_depth=1, random_state=0)
    reg = reg.fit(X, y)
    
    test = pd.read_csv('Data/test_set_VU_DM.csv')
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
   
# feature_importance()
# df = pd.read_csv('Data/test_data.csv')
# add_values(df, 'Data/test_data.csv')
# df = pd.read_csv('Data/train_data.csv')
# add_values(df, 'Data/train_data.csv')
# test_model(train_model())
# drop_nan_columns()
# check_same_values()
# make_files()

