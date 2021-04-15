import pandas as pd
import Cleaner
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 18})


def make_submission(train, test, clf, name):
    features = ['Title_num', 'Sex_2', 'Fare', 'Age', 'Sex_1', 'Pclass_1','Pclass_2','Pclass_3', 'Family_Size']
    x = train[features]
    y = train["Survived"]  
    clf = clf.fit(x, y)

    plot_confusion_matrix(clf, x, y)
    plt.tight_layout()
    plt.show()  

    x = test[features]
    predictions = clf.predict(x)

    dataframe = pd.DataFrame()
    dataframe["PassengerId"] = test["PassengerId"]
    dataframe["Survived"] = predictions
    dataframe["Survived"] = dataframe["Survived"].astype(int)
    
    # print(list(predictions).count(1))
    # dataframe.to_csv(f"Competition_{name}.csv", index=False)
    # print(predictions)



