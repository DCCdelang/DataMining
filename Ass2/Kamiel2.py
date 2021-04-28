import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

df = pd.read_csv('Data/splitted.csv')
for i in df['position']:
    print(i)

# df = df.fillna(0)
# x = list(df)[3:-1]
# X = df[x]
# y = df["values"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size=0.2, random_state=11)


# clf = RandomForestRegressor(max_depth=2, random_state=0)
# y_pred = clf.fit(X_train, y_train).predict(X_test)
# print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
# print(y_test)

# print(y_test, y_pred)
# for count, i in enumerate(y_pred):
#     if i != 0:
#         print(y_test[count])
# print(clf.score(X_test, y_test))

