import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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


   validation = []
   for i in range(1000):

      l =np.random.randint(1,1000000)
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
      validation.append(val_mae)
   d = {'col1': validation}

   df1 = pd.DataFrame(data=d)


   sns.histplot(data=df1, x="col1")
   plt.xlabel("Mae", fontsize=14)
   plt.ylabel("Count", fontsize=14)
   plt.xticks(fontsize=14)
   plt.yticks(fontsize=14)
   # plt.legend(fontsize=14)
   plt.tight_layout()
   plt.savefig("Tree_distribution.pdf")
   plt.show()
   print(np.mean(validation))
   # print(val_mae)
   # print(cross_val_score(model, x, y, cv=10))

def forest(df, features, y):
   
   x = df[features]
   y = df[y]
   validation = []
   for i in range(1000):
      l =np.random.randint(1,1000000)


      model = RandomForestRegressor(criterion='mae',random_state=1, n_estimators=100)

      train_X, val_X, train_y, val_y = train_test_split(x, y, random_state=l,test_size=0.2, train_size=0.8)

      # fit your model
      model.fit(train_X, train_y)
      
      predicts = model.predict(val_X)
 
   #       good += 1
   #    else:
   #       wrong +=1
      
      # print(i)
      # print()
   # Calculate the mean absolute error of your Random Forest model on the validation data
   val_mae = mean_absolute_error(predicts, val_y)
   print('forest mae', val_mae)
   print('forest, r2;', metrics.r2_score(val_y, predicts))
   #print(cross_val_score(model, x, y, cv=10))
   
   
def forest_2(df, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=test_size) #random_state=seed
    
    rf = RandomForestRegressor(n_estimators = 1000)
    rf.fit(X_train, y_train)
    
    predicts = rf.predict(X_test)
    val_mae = mean_absolute_error(predicts, y_test)
    
    print(metrics.r2_score(y_test, predicts))
    
    print(val_mae)
    #print(cross_val_score(rf, df, y, cv=10))
    
    #return(val_mae)
    


def bayes(df, features, y):
   x = df[features]
   y = df[y]
   y=y.astype('int')
   #Create a Gaussian Classifier
   gnb = GaussianNB()
   validation = []
   for i in range(1000):   
      l =np.random.randint(1,1000000)
      train_X, val_X, train_y, val_y = train_test_split(x, y, random_state=l,test_size=0.2, train_size=0.8)

      #Train the model using the training sets
      gnb.fit(train_X, train_y)

      #Predict the response for test dataset
      predicts = gnb.predict(val_X)


      # print(good, wrong)
      val_mae = mean_absolute_error(predicts, val_y)
      validation.append(val_mae)

   d = {'col1': validation}

   df1 = pd.DataFrame(data=d)


   sns.histplot(data=df1, x="col1")
   plt.xlabel("Mae", fontsize=14)
   plt.ylabel("Count", fontsize=14)
   plt.xticks(fontsize=14)
   plt.yticks(fontsize=14)
   # plt.legend(fontsize=14)
   plt.tight_layout()
   plt.savefig("Bayes_distribution.pdf")
   plt.show()
   print(np.mean(validation))
      
