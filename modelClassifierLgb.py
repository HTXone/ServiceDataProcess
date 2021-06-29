import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

max_feature = 50

train_data_file = './dataset/train_data002.csv'
# test_data_file = './val_data02.csv'
train_data = pd.read_csv(train_data_file, encoding='utf-8')
# test_data = pd.read_csv(test_data_file, encoding='utf-8')

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', 10)
train_data.head()
#print(train_data.shape)
#print(test_data.shape)

features_columns = [col for col in train_data.columns if col not in ['pax_name', 'pax_passport', 'emd_lable2','emd_lable']]
#features_columns_test = [col for col in test_data.columns ]

train = train_data[features_columns].values
target =train_data['emd_lable2'].values

#需要提交的测试集
#test = test_data[features_columns].values
#target_test = test_data['emd_lable2'].values

from sklearn.model_selection import train_test_split
import lightgbm
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.4, random_state=0)
X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=0)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
#print(test.shape, target_test.shape)

clf = lightgbm

train_matrix = clf.Dataset(X_train, label=y_train)
test_matrix = clf.Dataset(X_test, label=y_test)
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
early_stopping_rounds = 100
model = clf.train(params,
                  train_matrix,
                  num_round,
                  valid_sets=test_matrix,
                  early_stopping_rounds=early_stopping_rounds)



plt.figure(figsize=(12, 6))
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




SList = combine(List, FIList);
SList.sort(key=lambda x: x[1], reverse=True);



# pre= model.predict(X_valid,num_iteration=model.best_iteration)
#print(pre)

# print('score : ', np.mean((pre[:,1]>0.50)==y_valid))