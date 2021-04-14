import pandas as pd
import Cleaner
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler


def make_submission(train, test, clf):
    features = ["Pclass","PassengerId","Title_num","Binary_Sex","Family_Size","Age_div","Fare","Is_alone"]
    x = train[features]
    y = train["Survived"]  
    clf = clf.fit(x, y)


    x = test[features]
    predictions = clf.predict(x)

    dataframe = pd.DataFrame()
    dataframe["PassengerId"] = test["PassengerId"]
    dataframe["Survived"] = predictions
    
    dataframe.to_csv("Competition_SVC_2.csv", index=False)
    print(predictions)



