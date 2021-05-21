import pandas as pd
import random
import numpy as np
import seaborn as sns
import operator
import matplotlib.pyplot as plt
import matplotlib
from sklearn.inspection import permutation_importance
from sklearn.metrics import ndcg_score, make_scorer, SCORERS, accuracy_score, classification_report, precision_score,label_ranking_average_precision_score, make_scorer
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier, StackingRegressor, AdaBoostRegressor, GradientBoostingRegressor,GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.model_selection import GridSearchCV 
# from LambdaRankNN import LambdaRankNN
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.impute import KNNImputer
from scipy import stats
matplotlib.rcParams.update({'font.size': 18})
  
# train = pd.read_csv('Data/validation_test.csv')
# imputer = KNNImputer(n_neighbors=2)
# print('start imputing')
# train.loc[:, train.columns == 'prop_location_score2'] = imputer.fit_transform(train.loc[:, train.columns == 'prop_location_score2'])
# train.to_csv('test_validation_KNN.csv')

test = pd.read_csv('Data/test_set_VU_DM.csv')
train = pd.read_csv('Data/training_set_VU_DM.csv')
train = train.drop(columns=['click_bool', 'position', 'booking_bool', 'gross_bookings_usd'])
new = train.head(1)
new['prop_review_score'] = 0.5
total = pd.concat([test, train,new])
#
# print(stats.kstest(list(total['prop_starrating']), 'expon'))

# train['prop_review_score'] = round(train['prop_review_score'])
train["prop_review_score"] = pd.to_numeric(train["prop_review_score"],downcast='integer')

sns.countplot(x='prop_review_score',data=total, color = 'midnightblue')
# plt.xticks = ['0', '1', '2', '3', '4', '5']
plt.tight_layout()
plt.savefig('distribution_prop_review_score.pdf')
plt.show()

# # print(total[total['prop_location_score2']==1].shape)
# # print(total[total['prop_location_score2']==0].shape)
# # df = train
# # df_clicked =df[df['click_bool'] == 1]
# # df_new1 = df_clicked[['prop_id', 'random_bool', 'booking_bool', 'position']]
# # plt.figure(figsize=(15, 6))
# # sns.countplot(x = 'position', data = df_new1[df_new1.random_bool == 1], hue = 'booking_bool')
# # plt.tight_layout()
# # plt.xticks(np.arange(0, 40, 4))
# # # plt.yticks(np.arange(0, 2))
# # plt.title('Clicked and booked: Random order')
# # plt.legend(loc='upper right')


# # plt.savefig('Position_bias_random.pdf')
# # plt.show()
# # # train1 = train
# # # train1 = train1.drop(['booking_bool', 'click_bool', 'position', 'gross_bookings_usd'],axis=1)

# # # total = pd.read_csv('Data/concated.csv')
# # # plt.figure(figsize=(15, 6))
# # sns.countplot(x = 'position', data = df[(df.click_bool == 1) & (df.random_bool == 0) ], hue = 'booking_bool')

# # plt.title('Clicked and booked: Expedia order')

# # plt.legend(loc='upper right')
# # plt.xticks(np.arange(0, 41, 4))
# # plt.tight_layout()

# # plt.savefig('Position bias expedia.pdf')
# plt.show()
count_50 = 0
count_60 = 0
count_70 = 0
count_80 = 0
count_90 = 0


for i in total.columns:
    nan = total[i].isnull().sum(axis = 0)
    percentage = nan/len(total[i])
    
    if percentage < 0.5:
        print('\n')
        print(i)
        print("\n")
        print(total[i].describe())
        print("Nan = ",nan, 'Percentage = ', percentage)
        print("bool 1 percentage", len(total.loc[total[i] == 1])/ len(total[i]))
    if percentage > 0.50:

        count_50 +=1
    if percentage >0.60:
        count_60 +=1
    if percentage > 0.70:
        count_70 +=1
    if percentage > 0.80:
        count_80 += 1
    if percentage > 0.90:
        count_90 +=1


print(count_50,count_60,count_70,count_80,count_90)



