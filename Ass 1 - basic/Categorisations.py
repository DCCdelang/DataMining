import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score


from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_validate
from sklearn import metrics



def tree(df, features, y):
   x = df[features]
   y = df[y]
   l =np.random.randint(1,100)
   train_X, val_X, train_y, val_y = train_test_split(x, y, random_state=l,test_size=0.2, train_size=0.8)

   model = DecisionTreeRegressor(criterion='mae',random_state=1)

   model_f = model.fit(train_X, train_y)
   
   predicts = model_f.predict(val_X)
   
   # good = 0
   # wrong = 0

   # for count, i in enumerate(predicts):
   #    j = val_y.iloc[count]
   #    if float(i)>0 and float(i)<34:
   #       a = "low"
   #    elif float(i) > 33 and float(i) < 67:
   #       a = "med"
   #    else:
   #       a = "high"
         

   #    if float(j)>0 and float(j)<34:
   #       b = "low"
   #    elif float(j) > 33 and float(j) < 67:
   #       b = "med"
   #    else:
   #       b = "high"
   #    if a==b:
   #       good+=1
   #    else:
   #       wrong+=1

   # for count, i in enumerate(predicts):
   #    if (list(i).index(max(list(i)))) == (list(val_y.iloc[count]).index(max(list(val_y.iloc[count])))):
 
   #       good += 1
   #    else:
   #       wrong +=1

   # print(good, wrong)

   val_mae = mean_absolute_error(predicts, val_y)

   print(val_mae)
   print(cross_val_score(model, x, y, cv=10))

def forest(df, features, y):
   
   x = df[features]
   y = df[y]
   l =np.random.randint(1,100)
   model = RandomForestRegressor(criterion='mae',random_state=1, n_estimators=10)

   train_X, val_X, train_y, val_y = train_test_split(x, y, random_state=l,test_size=0.2, train_size=0.8)

   # fit your model
   model.fit(train_X, train_y)
   
   predicts = model.predict(val_X)
   good = 0
   wrong = 0


      

   # for count, i in enumerate(predicts):
     
   #    if (list(i).index(max(list(i)))) == (list(val_y.iloc[count]).index(max(list(val_y.iloc[count])))):
 
   #       good += 1
   #    else:
   #       wrong +=1
      
      # print(i)
      # print()
   # Calculate the mean absolute error of your Random Forest model on the validation data
   val_mae = mean_absolute_error(predicts, val_y)
   print('forest mae', val_mae)
   print('forest, r2;', metrics.r2_score(val_y, predicts))
   print(cross_val_score(model, x, y, cv=10))
   
   
def forest_2(df, y, test_size, seed = 43, fold = 10):
    pipeline = Pipeline([

        ('regressor', RandomForestRegressor(random_state = seed,n_estimators = 1000 ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=test_size) #random_state=seed
    
    pipeline.fit(X_train, y_train)
    predicts = pipeline.predict(X_test)
    
    mae = mean_absolute_error(predicts, y_test)
    #mape = np.mean(np.abs((y_test - predicts) / np.abs(y_test)))
    #print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
    #print('Accuracy:', round(100*(1 - mape), 2))
    
    print('mae', mae)
    print('r2',metrics.r2_score(y_test, predicts))
    
    #choose metrics to score the fit
    scoring = ['r2', 'neg_mean_absolute_error']
    
    #cross validation (10 fold)
    cross_val = cross_validate(pipeline['regressor'], df, y, cv = fold, scoring=scoring)
    
    r2 = cross_val['test_r2']
    neg_mae = cross_val['test_neg_mean_absolute_error']
    print('Fold is', fold)
    print()
    print('r2 mean', r2.mean())
    print('r2 std', r2.std())
    
    print()
    print('neg mae mean', neg_mae.mean())
    print('neg mae std', neg_mae.std())


def bayes(df, featuress, y):
   x = df[features]
   y = df[y]
   y=y.astype('int')
   #Create a Gaussian Classifier
   gnb = GaussianNB()

   l =np.random.randint(1,100)
   train_X, val_X, train_y, val_y = train_test_split(x, y, random_state=l,test_size=0.2, train_size=0.8)

   #Train the model using the training sets
   gnb.fit(train_X, train_y)

   #Predict the response for test dataset
   predicts = gnb.predict(val_X)


   # print(good, wrong)
   val_mae = mean_absolute_error(predicts, val_y)
   print(val_mae)
   print(cross_val_score(gnb, x, y, cv=10))
