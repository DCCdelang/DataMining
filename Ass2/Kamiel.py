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
    train = pd.read_csv('Data/validation_train_1.csv')
    train = train.fillna(0)

    features = list(train.columns)
    features.remove('value')
    features.remove('srch_id')
    X = train[features]
    y = train["value"]  
    # estimators = [
    # ('rf', RandomForestRegressor(n_estimators = 50, random_state=0))]
   
    # reg = StackingRegressor(
    # estimators=estimators,
    # final_estimator=GradientBoostingRegressor(n_estimators=50, learning_rate=1.0, max_depth=2, random_state=0))
    reg = GradientBoostingRegressor(n_estimators=50, learning_rate=1.0, max_depth=2, random_state=0)
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
    
    test = pd.read_csv('Data/validation_test_1.csv')
    
    test = test.fillna(0)
    scores = []
    ids = list(set(test['ra']))
    features = list(test.columns)
    
    features.remove('value')
    features.remove('ra')
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
    train = pd.read_csv('Data/clicked_data_1.csv')
    train = train.fillna(0)
    features = list(train.columns)
    features.remove('value')
    
    X = train[features]
    y = train["value"]

    estimators = [
    ('rf', RandomForestRegressor(n_estimators = 50, random_state=0))]
   
    reg = StackingRegressor(
    estimators=estimators,
    final_estimator=GradientBoostingRegressor(n_estimators=10, learning_rate=1.0, max_depth=2, random_state=0))
    reg = reg.fit(X, y)
    
    test = pd.read_csv('Data/processed_test_1.csv')
    test = test.fillna(0)
    df_sub = test
    
    X = test[features] 
    predictions = reg.predict(X)

    df_sub['predicted_values'] = predictions

    df_sub = df_sub.sort_values(['srch_id', 'predicted_values'], ascending=[True, False])
    
    df_sub = df_sub[['srch_id', 'prop_id']]

    df_sub.to_csv('Data/submission1.csv', index=False)


    # print(df.columns,df1.columns)

# feature_importance()

# make_clicked_file()

# print("1")
# df = pd.read_csv('Data/processed_train.csv')
# df = make_files(df, 'test')
# df.to_csv('Data/validation_test.csv', index=False)

# print("2")
# df = pd.read_csv('Data/clicked_data.csv')
# df = make_files(df,'train')
# df.to_csv('Data/validation_train.csv', index=False)

# print("2")
# df = pd.read_csv('Data/validation_train.csv')
# add_values(df, 'Data/validation_train.csv')

# print("2")
# df = pd.read_csv('Data/validation_test.csv')
# add_values(df, 'Data/validation_test.csv')

# df = pd.read_csv('Data/validation_test.csv')
# print(df.shape)
# df = drop_columns(df)
# df.to_csv('Data/validation_test_1.csv', index=False)

# df = pd.read_csv('Data/validation_train.csv')
# print(df.shape)
# df = drop_columns(df)
# df.to_csv('Data/validation_train_1.csv', index=False)

# test_model(train_model())

# drop_nan_columns()

# check_same_values()


# test_model(train_model())
# random()

# df = pd.read_csv('Data/processed_train.csv')
# print(df.shape)
# df = drop_columns(df)
# df.to_csv('Data/validation_train.csv', index=False)

# print(df.shape)

# df = pd.read_csv('Data/validation_test.csv')
# df = drop_columns(df)
# df.to_csv('Data/validation_test.csv', index=False)


# print(df.shape)

# df = pd.read_csv('Data/clicked_data.csv')
# add_values(df, 'Data/clicked_data.csv')

df = pd.read_csv('Data/clicked_data.csv')
print(df.shape)
df = drop_columns(df)
df.to_csv('Data/clicked_data_1.csv', index=False)

print(df.shape)

df = pd.read_csv('Data/processed_test.csv')
print(df.shape)
df = drop_columns(df)
df.to_csv('Data/processed_test_1.csv', index=False)
print(df.shape)

make_submission_file()