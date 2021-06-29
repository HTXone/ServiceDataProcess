import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

train_data_file = './dataset/ShowTrain.csv'
test_data_file = './dataset/ShowTest.csv'
train_data = pd.read_csv(train_data_file, encoding='utf-8')
test_data = pd.read_csv(test_data_file, encoding='utf-8')

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', 10)
print(train_data.shape)
print(test_data.shape)
train_data.head()

features_columns = [col for col in train_data.columns if col not in ['pax_name', 'pax_passport', 'emd_lable', 'emd_lable2']]

train = train_data[features_columns].values
target =train_data['emd_lable2'].values

from sklearn.feature_selection import SelectKBest
from array import array
from sklearn.feature_selection import chi2

SKB = SelectKBest(chi2, k=80)
train = SKB.fit_transform(train, target)

print(train.shape)
features_List = SKB.get_support(indices=True)
for i in features_List:
    print(features_columns[i])

train = pd.DataFrame(train)
train.head(5)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

clf = RandomForestClassifier(n_estimators=500,
                             max_depth=70,
                             random_state=5,
                             n_jobs=-1,
                             #class_weight="balanced"
                             )
# from catboost import CatBoostClassifier
# clf = CatBoostClassifier(iterations=300,
#                 depth=12,
#                 random_seed=0,
#                 thread_count=-1,
#                 learning_rate = 0.4,
#                 task_type="GPU",
#                 loss_function = "CrossEntropy",
#                 #nan_mode= "Max"
#             )


X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.27, random_state=0)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#验证集
clf = clf.fit(X_train, y_train)
#clf = clf.fit(X_train, y_train,plot=True)
print('=======')
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print("modelscore:", clf.score(X_test, y_test))
print('=======')


#设定阈值
threshold = 0.39
predicted_proba = clf.predict_proba(X_test)
print(predicted_proba.shape)

#重新验证
predicted = (predicted_proba [:,1] >= threshold).astype('int')#
accuracy = accuracy_score(y_test, predicted)
print(classification_report(y_test, predicted))
print("accuracy: ", accuracy)

shape_test = X_test.shape[0]
print(shape_test)
pred_1d = predicted_proba[:,1].reshape(shape_test,)
y_test = y_test.reshape(shape_test,)
print("AUC: ",roc_auc_score(y_true, pred_1d))

#测试集
print(predicted.shape)
data1 = predicted_proba.reshape(shape_test,2)

def find_repeat_data(name_list):
    """
    查找列表中重复的数据
    :param name_list:
    :return: 一个重复数据的列表，列表中字典的key 是重复的数据，value 是重复的次数
    """
    repeat_list = []
    for i in set(name_list):
        ret=name_list.count(i) # 查找该数据在原列表中的个数
        if ret > 1:
            item=dict()
            item[i] = ret
            repeat_list.append(item)
    print(repeat_list)
    return repeat_list

find_repeat_data(list(y_test))
find_repeat_data(list(predicted))