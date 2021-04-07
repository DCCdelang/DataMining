import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def tree(df, features, y):
   x = df[features]
   y = df[y]

   train_X, val_X, train_y, val_y = train_test_split(x, y, random_state=1)

   model = DecisionTreeRegressor(random_state=1)

   model = model.fit(train_X, train_y)
   
   predicts = model.predict(val_X)
   good = 0
   wrong = 0
   for count, i in enumerate(predicts):
     
      if (list(i).index(max(list(i)))) == (list(val_y.iloc[count]).index(max(list(val_y.iloc[count])))):
 
         good += 1
      else:
         wrong +=1
   print(good, wrong)
   val_mae = mean_absolute_error(predicts, val_y)

   print(val_mae)

def forest(df, features, y):
   
   x = df[features]
   y = df[y]
   model = RandomForestRegressor(random_state=1)

   train_X, val_X, train_y, val_y = train_test_split(x, y, random_state=1)

   # fit your model
   model.fit(train_X, train_y)
   
   predicts = model.predict(val_X)
   good = 0
   wrong = 0
   for count, i in enumerate(predicts):
     
      if (list(i).index(max(list(i)))) == (list(val_y.iloc[count]).index(max(list(val_y.iloc[count])))):
 
         good += 1
      else:
         wrong +=1
   print(good, wrong)
      
      
      # print(i)
      # print()
   # Calculate the mean absolute error of your Random Forest model on the validation data
   val_mae = mean_absolute_error(predicts, val_y)
   print("Validation MAE for Random Forest Model: {}".format(val_mae))