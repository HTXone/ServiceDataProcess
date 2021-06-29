import numpy as np # linear algebra
import pandas as pd
import seaborn as sns
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn import metrics
from scipy.optimize import curve_fit
# from scipy import asarray as ar,exp
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve

from tqdm import tqdm_notebook as tqdm

train_data_file = './dataset/trainFinal5.csv'
test_data_file = './dataset/ValFinal5.csv'
train_data = pd.read_csv(train_data_file, encoding='utf-8')
test_data = pd.read_csv(test_data_file, encoding='utf-8')


pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', 10)
train_data.head()
#print(train_data.shape)
#print(test_data.shape)
features_columns = [col for col in train_data.columns if col not in ['pax_name', 'pax_passport', 'emd_lable2', 'emd_lable']]
#features_columns_test = [col for col in test_data.columns ]

train = train_data[features_columns].values
test = test_data[features_columns].values

target =train_data['emd_lable2'].values
target_test = test_data['emd_lable2'].values

# tr_te = train
# for col in tqdm(train):
#     if 'age' in col:
#         tr_te[col + '_cnt'] = tr_te[col].map(tr_te[col].value_counts())

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from test import LR

#GBDT作为基模型的特征选择
# train = SelectFromModel(GradientBoostingClassifier()).fit_transform(train, target)

#LR特征选择
# train = SelectFromModel(LR(threshold=0.6, C=0.1)).fit_transform(train, target)


from sklearn.feature_selection import SelectKBest
from minepy import MINE
from array import array
from sklearn.feature_selection import chi2

# 由于MINE的设计不是函数式的，定义mic方法将其为函数式的，返回一个二元组，二元组的第2项设置成固定的P值0.5
def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return (m.mic(), 0.5)
#
# train = SelectKBest(chi2, k=120).fit_transform(train, target)

# 选择K个最好的特征，返回特征选择后的数据
# train = SelectKBest(lambda X, Y: np.array(list(map(lambda x:mic(x, Y), X.T))).T, k=100).fit_transform(train, target)
SKB = SelectKBest(chi2, k=80)
train = SKB.fit_transform(train, target)
test = SKB.transform(test)
print(train.shape)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

clf = RandomForestClassifier(n_estimators=500,
                             max_depth=70,
                             random_state=0,
                             n_jobs=-1)

X_train, X_test, y_train, y_test = train, test,target,target_test

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)



clf = clf.fit(X_train, y_train)


print('=======')
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print("RandomForest:", clf.score(X_test, y_test))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print('=======')
# print(y_pred[:10])

#print(clf.predict(X_test))

#设定阈值
threshold = 0.53

predicted_proba = clf.predict_proba(X_test)
print(predicted_proba.shape)
# print(predicted_proba)
predicted = (predicted_proba [:,1] >= threshold).astype('int')#a
print(np.sum(predicted))
accuracy = accuracy_score(y_test, predicted)
print(accuracy)
print(predicted.shape)
#predicted.reshape(9373,2)
data1 = predicted_proba.reshape(X_test.shape[0],2)

#print(train.shape, target.shape)
#print(test.shape, target_test.shape)
"""
#测试集
print("test:")
clf.score(test, target_test)

print(clf.predict(test))
#data1 = clf.predict(test)
#print(clf.predict_proba(test))
data2 = clf.predict_proba(test)
data2.reshape(6771,2)"""


# def find_repeat_data(name_list):
#     """
#     查找列表中重复的数据
#     :param name_list:
#     :return: 一个重复数据的列表，列表中字典的key 是重复的数据，value 是重复的次数
#     """
#     repeat_list = []
#     for i in set(name_list):
#         ret=name_list.count(i) # 查找该数据在原列表中的个数
#         if ret > 1:
#             item=dict()
#             item[i] = ret
#             repeat_list.append(item)
#     print(repeat_list)
#     return repeat_list
#
# find_repeat_data(list(y_test))
# find_repeat_data(list(predicted))
# #find_repeat_data(list(target_test))
#
#
# print(type(data1))
# #print(data2)
# df = pd.DataFrame(data1)
# df.insert(2, 'answer', list(y_test))  # 2表示插入列的位置（索引）， 'answer'是列标题
# print(type(df))
# print(df)
#df.to_csv('output_classifier.csv')

