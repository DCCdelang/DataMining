import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score


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
   print(val_mae)
   print(cross_val_score(model, x, y, cv=10))


def bayes(df, features, y):
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
