import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def add_values(df):
    
    df['value'] = df.apply(lambda row: row.click_bool + (row.booking_bool * 4), axis=1)
    return df

data = pd.read_csv("Data/fifty_fifty.csv")
data = data.fillna(0)

data = add_values(data)
# data[data < 0] = 0
features = list(data.columns)
print(len(features))
features.remove('value')
features.remove('click_bool')
features.remove('gross_bookings_usd')
features.remove('booking_bool')
features.remove('srch_id')
features.remove('position')


X = data[features] #independent columns
y = data['value']    #target column i.e price range#apply SelectKBest class to extract top 10 best features

bestfeatures = SelectKBest(score_func=f_classif, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
df = featureScores.nlargest(25,'Score')
print(featureScores.nlargest(25,'Score'))  #print 10 best features


model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()


# get correlations of each features in dataset
print(list(df['Specs']))
new_features = list(df['Specs'])
new_features.append('value')

data = data[new_features]
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")

plt.show()

