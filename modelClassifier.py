import pandas as pd
import numpy as np
import random
import tqdm
from PIL import Image
from matplotlib import pyplot as plt

from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
from sklearn.model_selection import KFold # import KFold

train_data_file = './dataset/trainFinal2.csv'
test_data_file = './dataset/val_Final_data.csv'
train_data = pd.read_csv(train_data_file, encoding='utf-8')
test_data = pd.read_csv(test_data_file, encoding='utf-8')

str2 = """pax_fcny
pax_tax
tkt_3y_amt
dist_cnt_y1
dist_cnt_y3
avg_dist_cnt_y2
tkt_avg_amt_y3
ffp_nbr
avg_dist_cnt_y3
dist_i_cnt_y3
dist_i_cnt_y2
tkt_avg_amt_y2
dist_all_cnt_y2
flt_bag_cnt_y3
tkt_all_amt_m3
prebuy_d_cnt_m3_d99
select_seat_cnt_y2
seat_middle_cnt_y3
tkt_all_amt_y3
flt_bag_cnt_y2
prebuy_i_cnt_y3_d99
seat_walkway_cnt_y2
pit_avg_amt_y3
seat_window_cnt_y3
pit_accu_air_amt
flt_delay_time_m3
tkt_i_amt_y2"""

LL = str2.split("\n")

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', 10)
train_data.head()
#print(train_data.shape)
#print(test_data.shape)

# while(1):
features_columns = [col for col in train_data.columns if
                    col not in ['pax_name', 'pax_passport', 'emd_lable2', 'emd_lable']]
# features_columns_test = [col for col in test_data.columns ]
# features_columns = random.sample(features_columns, 50)
# print(features_columns)
# train = train_data[features_columns]
#
# import test
# for col in features_columns :
#     TempList = train[col]
#     rgn = test.RGN(TempList)
#     train[col] = rgn.output

train = train_data[features_columns].values
test = test_data[features_columns].values

target = train_data['emd_lable2'].values
ID = test_data['pax_passport'].values
# target2 = train_data['emd_lable2'].values

print(train.shape)

from sklearn.feature_selection import SelectKBest
from minepy import MINE
from array import array
from sklearn.feature_selection import chi2

SB = SelectKBest(chi2, k=71)
train = SB.fit_transform(train, target)
print(SB.get_support(indices = True))
test = test[:,SB.get_support(indices=True)]
print(train.shape,test.shape)

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from test import LR

# GBDT作为基模型的特征选择
# from sklearn.ensemble import RandomForestClassifier as RFC
# RFC_ = RFC(n_estimators =50,random_state=0)
# train = SelectFromModel(RFC_,threshold=0.005).fit_transform(train, target)

#LR特征选择
# train = SelectFromModel(LR(threshold=0.6, C=0.1)).fit_transform(train, target)

from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import PCA
#线性判别分析法，返回降维后的数据
#参数n_components为降维后的维数
# pca_sk = PCA(n_components=20)
# train = pca_sk.fit_transform(train)
print(train.shape)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

# clf = CatBoostClassifier(
#                 iterations=300,
#                 depth=12,
#                 random_seed=1,
#                 thread_count=-1,
#                 learning_rate = 1,
#                 task_type="GPU",
#                 # loss_function='Logloss',
#                 # n_estimators=1000,
#                 # devices='0:1',
#                 loss_function = "CrossEntropy",
#                 custom_metric = "F1",
#                 leaf_estimation_method = "Newton",
#             )

clf = RandomForestClassifier(n_estimators=500,
                             max_depth=70,
                             random_state=0,
                             n_jobs=-1)

'''
print(sorted(Counter(y_train).items()))

# smo = SMOTE()
adasyn = ADASYN()
X_train_smote,y_train_smote =  adasyn.fit_resample(X_train, y_train)
print(sorted(Counter(y_train_smote).items()))
'''
#


# from sklearn.linear_model import LogisticRegression
# from imblearn.under_sampling import InstanceHardnessThreshold
# iht = InstanceHardnessThreshold(random_state=0,
#                                 estimator=LogisticRegression())
# X_resampled, y_resampled = iht.fit_resample(X_train, y_train)
# print(X_resampled.shape)

# 验证集

#
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.3, random_state=0)
#
# from imblearn.combine import SMOTETomek
# smote_tomek = SMOTETomek(random_state=0)
# X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

print(X_train.shape, y_train.shape)
# print(X_resampled.shape, y_resampled.shape)
print(X_test.shape, y_test.shape)


# import pandas as pd # 数据科学计算工具
# import numpy as np # 数值计算工具
# import matplotlib.pyplot as plt # 可视化
# import seaborn as sns # matplotlib的高级API
# from sklearn.model_selection import StratifiedKFold #交叉验证
# from sklearn.model_selection import GridSearchCV #网格搜索
# from sklearn.model_selection import train_test_split #将数据集分开成训练集和测试集
# from xgboost import XGBClassifier                     #xgboost
#
#
# model = XGBClassifier()
# learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]  # 学习率
# gamma = [1, 0.1, 0.01, 0.001]
#
# param_grid = dict(learning_rate=learning_rate, gamma=gamma)  # 转化为字典格式，网络搜索要求
#
# kflod = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)  # 将训练/测试数据集划分10个互斥子集，
#
# grid_search = GridSearchCV(model, param_grid, scoring='neg_log_loss', n_jobs=-1, cv=kflod)
# # scoring指定损失函数类型，n_jobs指定全部cpu跑，cv指定交叉验证
# grid_result = grid_search.fit(X_train, y_train)  # 运行网格搜索
# print("Best: %f using %s" % (grid_result.best_score_, grid_search.best_params_))
# # grid_scores_：给出不同参数情况下的评价结果。best_params_：描述了已取得最佳结果的参数的组合
# # best_score_：成员提供优化过程期间观察到的最好的评分
# # 具有键作为列标题和值作为列的dict，可以导入到DataFrame中。
# # 注意，“params”键用于存储所有参数候选项的参数设置列表。
# means = grid_result.cv_results_['mean_test_score']
# params = grid_result.cv_results_['params']
# for mean, param in zip(means, params):
#     print("%f  with:   %r" % (mean, param))



clf = clf.fit(X_train, y_train)
# clf = clf.fit(train, target)
# clf = clf.fit(X_resampled, y_resampled)

print('=======')
y_true, y_pred = y_train, clf.predict(X_train)
print(classification_report(y_true, y_pred))
print("RandomForest:", clf.score(X_train, y_train))
print("MAE:", mean_absolute_error(y_train, y_pred))
print("MSE:", mean_squared_error(y_train, y_pred))
print('=======')

print('\n\n=======')
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print("RandomForest:", clf.score(X_test, y_test))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print('=======')

# print(y_pred[:10])

# print(clf.predict(X_test))

# 设定阈值
threshold = 0.39

predicted_proba = clf.predict_proba(X_test)
print(predicted_proba.shape)
# print(predicted_proba)

Sum = 0
for i in range(len(y_test)):
    if(y_test[i] == 1 and predicted_proba[i,1]>threshold) :
        Sum+=1
print('Sum:{}'.format(Sum))
predicted = (predicted_proba[:, 1] >= threshold).astype('int')  # a
print(np.sum(predicted))
accuracy = accuracy_score(y_test, predicted)
print(accuracy)
print("AUC: ",roc_auc_score(y_test, predicted))
print(classification_report(y_test, predicted))

print(predicted.shape)


predicted_proba_Val = clf.predict_proba(test)
# print(predicted_proba_Val.shape)
ID = ID.reshape(ID.shape[0],1)
# print(ID.shape)
predicted_proba_Val = np.hstack((predicted_proba_Val,ID))
print(predicted_proba_Val.shape)


Ans = pd.DataFrame(predicted_proba_Val)
Ans.to_csv('./dataset/temp.csv')