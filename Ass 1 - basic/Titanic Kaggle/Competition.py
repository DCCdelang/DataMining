import pandas as pd
import Cleaner
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler
def make_submission(clf):
    df = pd.read_csv("Data/test.csv")
    df = pd.read_csv("Data/train.csv")

    df = Cleaner.Class(df)
    df = Cleaner.Binary_Sex(df)
    df = Cleaner.Binary_cabin(df)
    df = Cleaner.SexClass(df)
    df = Cleaner.family_size(df)
    df = Cleaner.fill_age(df)
    df = Cleaner.age_class(df)
    df = Cleaner.AgeClass(df)
    df = Cleaner.replace_titles(df)
    df = Cleaner.title_num(df)

    df =Cleaner.family_size(df)
    df = Cleaner.is_alone(df)
    x = df[["Pclass","PassengerId","Title_num","Binary_Sex","Family_Size","Age_div","Fare","Is_alone"]]
    y = df["Survived"]  

  

    clf = clf.fit(x, y)

    df = pd.read_csv("Data/test.csv")
    df = Cleaner.Class(df)
    df = Cleaner.Binary_Sex(df)
    df = Cleaner.Binary_cabin(df)
    df = Cleaner.SexClass(df)
    df = Cleaner.family_size(df)
    df = Cleaner.fill_age(df)
    df = Cleaner.age_class(df)
    df = Cleaner.AgeClass(df)
    df = Cleaner.replace_titles(df)
    df = Cleaner.title_num(df)
    print(df["PassengerId"])
    df =Cleaner.family_size(df)
    df = Cleaner.is_alone(df)
    features = ["Pclass","PassengerId","Title_num","Binary_Sex","Family_Size","Age_div","Fare","Is_alone"]
    df['Fare'] = df['Fare'].fillna(0)
    x = df[features]
    
    predictions = clf.predict(x)

    dataframe = pd.DataFrame()
    dataframe["PassengerId"] = df["PassengerId"]
    dataframe["Survived"] = predictions
    
    dataframe.to_csv("Competition.csv", index=False)
    print(predictions)


pipeline = Pipeline([('scale', StandardScaler()),
    ('classifier', RandomForestClassifier(criterion="entropy",  n_estimators=100, min_samples_leaf=2, max_depth=None, random_state=0))
])

make_submission(pipeline)