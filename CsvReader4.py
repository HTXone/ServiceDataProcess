import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series,DataFrame

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import OrdinalEncoder

# from category_encoders import *

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
train_data_file = './dataset/trainFinal.csv'

train_data = pd.read_csv(train_data_file, encoding='utf-8')


pd.set_option('display.max_columns', None)

print(train_data.shape)


train_data.fillna(0, inplace=True) #填充

train_data.head()

max_feature = 150
features_columns = [col for col in train_data.columns if col not in ['pax_name', 'pax_passport', 'emd_lable2','emd_lable']]    #直接使用list读取数据
target = train_data['emd_lable2'].values
# print(features_columns)
train = train_data[features_columns]


ordinal_encoder = OrdinalEncoder()

text_encoded = ordinal_encoder.fit_transform(train)

train_data[features_columns] = text_encoded


from sklearn.model_selection import train_test_split
import lightgbm
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

#print(test.shape, target_test.shape)

clf = lightgbm

train_matrix = clf.Dataset(text_encoded[0:23432], label=target[0:23432])
test_matrix = clf.Dataset(text_encoded[0:23432], label=target[0:23432])
params = {
          'boosting_type': 'gbdt',
          #'boosting_type': 'dart',
          'objective': 'multiclass',
          'metric': 'multi_logloss',
          #'metric': 'auc',
          'min_child_weight': 1.5,
          'max_depth': 10,         #树的最大深度
          'num_leaves': 40,        #取值应 <= 2 ^（max_depth）， 超过此值会导致过拟合
          'lambda_l2': 10,
          'subsample': 0.7,
          'colsample_bytree': 0.7,
          'colsample_bylevel': 0.7,
          'learning_rate': 0.01,   #学习率
          'tree_method': 'exact',
          'seed': 2017,
          "num_class": 2,          #分类数
          'silent': True,         #静默模式开启，不会输出任何信息。
          }
num_round = 3600
early_stopping_rounds = 150
model = clf.train(params,
                  train_matrix,
                  num_round,
                  valid_sets=test_matrix,
                  early_stopping_rounds=early_stopping_rounds)



plt.figure(figsize=(10, 6))
clf.plot_importance(model, max_num_features=max_feature)
plt.title("Featurertances")
plt.show()
FIList = model.feature_importance()
print(FIList)

def combine(outputList, sortList):
    CombineList = list()
    for index in range(len(outputList)):
        CombineList.append((outputList[index], sortList[index]))
    return CombineList


SList = combine(features_columns, FIList)
SList.sort(key=lambda x: x[1], reverse=True)
IndexList = []
for i in range(max_feature):
    IndexList.append(SList[i][0])

print(IndexList)

IndexList.append('pax_name')
IndexList.append('pax_passport')
IndexList.append('emd_lable2')
IndexList.append('emd_lable')

print(IndexList)

train = train_data[IndexList].ix[0:23432,:]
test = train_data[IndexList].ix[23432:,:]

train.to_csv("./dataset/Final_train.csv",index=False,encoding='utf_8_sig')
test.to_csv("./dataset/Final_val.csv",index=False,encoding='utf_8_sig')


