import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series,DataFrame

from mpl_toolkits.mplot3d import Axes3D





# Train_data = pd.read_csv('./dataset/train.csv')
# DF = pd.DataFrame(Train_data)
# for i in range(Train_data.shape[0]):
#     DF.loc[i,'seat'] = seat(DF.loc[i, 'seat_window_cnt_y3'],DF.loc[i,'seat_walkway_cnt_y3'],DF.loc[i,'seat_middle_cnt_y3'])
#     DF.loc[i,'cabin'] = cabin(DF.loc[i,'cabin_hf_cnt_y3'],DF.loc[i,'cabin_f_cnt_y3'],DF.loc[i,'cabin_hy_cnt_y3'],DF.loc[i,'cabin_y_cnt_y3'],DF.loc[i,'cabin_c_cnt_y3'])
#     DF.loc[i,'country1'] = country(DF.loc[i,'residence_country'])
#     DF.loc[i,'country2'] = country(DF.loc[i,'nation_name'])
# Train_data.to_csv('./dataset/train3.csv',encoding='utf_8_sig')


# train_data = pd.read_csv('./train_data01.csv', encoding='utf-8')

train_data_file = './dataset/train_originClear_02.csv'
# test_data_file = './val_data01.csv'
train_data = pd.read_csv(train_data_file, encoding='utf-8')
# test_data = pd.read_csv(test_data_file, encoding='utf-8')

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', 10)
#train_data.head(50)
print(train_data.shape)
# print(test_data.shape)


from sklearn.preprocessing import OrdinalEncoder

# train_data["seg_cabin"].fillna('Y', inplace=True) #填充
train_data["seg_cabin"].fillna('Y', inplace=True) #填充
# train_data["pit_add_chnl_y2"].fillna('0', inplace=True) #填充
# train_data["pit_add_chnl_y3"].fillna('0', inplace=True) #填充
# train_data["city_name"].fillna('0', inplace=True) #填充
# test_data["pit_add_chnl_y3"].fillna('0', inplace=True) #填充
#seg_cabin
str = """seg_route_to
seg_cabin
pax_fcny
pax_tax
gender
age
residence_country
nation_name
province_name
marital_stat
ffp_nbr
member_level
cabin_hf_cnt_m3
cabin_hf_cnt_m6
cabin_hf_cnt_y1
cabin_hf_cnt_y2
cabin_hf_cnt_y3
cabin_f_cnt_m3
cabin_f_cnt_m6
cabin_f_cnt_y1
cabin_f_cnt_y2
cabin_f_cnt_y3
cabin_hy_cnt_m3
cabin_hy_cnt_m6
cabin_hy_cnt_y1
cabin_hy_cnt_y2
cabin_hy_cnt_y3
cabin_y_cnt_m3
cabin_y_cnt_m6
cabin_y_cnt_y1
cabin_y_cnt_y2
cabin_y_cnt_y3
cabin_c_cnt_m3
cabin_c_cnt_m6
cabin_c_cnt_y1
cabin_c_cnt_y2
cabin_c_cnt_y3
seat_window_cnt_m3
seat_window_cnt_m6
seat_window_cnt_y1
seat_window_cnt_y2
seat_window_cnt_y3
seat_walkway_cnt_m3
seat_walkway_cnt_m6
seat_walkway_cnt_y1
seat_walkway_cnt_y2
seat_walkway_cnt_y3
seat_middle_cnt_m3
seat_middle_cnt_m6
seat_middle_cnt_y1
seat_middle_cnt_y2
seat_middle_cnt_y3
prebuy_d_cnt_m3_d3
prebuy_d_cnt_m3_d7
prebuy_d_cnt_m3_d14
prebuy_d_cnt_m3_d30
prebuy_d_cnt_m3_d99
prebuy_d_cnt_m6_d3
prebuy_d_cnt_m6_d7
prebuy_d_cnt_m6_d14
prebuy_d_cnt_m6_d30
prebuy_d_cnt_m6_d99
prebuy_d_cnt_y1_d3
prebuy_d_cnt_y1_d7
prebuy_d_cnt_y1_d14
prebuy_d_cnt_y1_d30
prebuy_d_cnt_y1_d99
prebuy_d_cnt_y2_d3
prebuy_d_cnt_y2_d7
prebuy_d_cnt_y2_d14
prebuy_d_cnt_y2_d30
prebuy_d_cnt_y2_d99
prebuy_d_cnt_y3_d3
prebuy_d_cnt_y3_d7
prebuy_d_cnt_y3_d14
prebuy_d_cnt_y3_d30
prebuy_d_cnt_y3_d99
prebuy_i_cnt_m3_d3
prebuy_i_cnt_m3_d7
prebuy_i_cnt_m3_d14
prebuy_i_cnt_m3_d30
prebuy_i_cnt_m3_d99
prebuy_i_cnt_m6_d3
prebuy_i_cnt_m6_d7
prebuy_i_cnt_m6_d14
prebuy_i_cnt_m6_d30
prebuy_i_cnt_m6_d99
prebuy_i_cnt_y1_d3
prebuy_i_cnt_y1_d7
prebuy_i_cnt_y1_d14
prebuy_i_cnt_y1_d30
prebuy_i_cnt_y1_d99
prebuy_i_cnt_y2_d3
prebuy_i_cnt_y2_d7
prebuy_i_cnt_y2_d14
prebuy_i_cnt_y2_d30
prebuy_i_cnt_y2_d99
prebuy_i_cnt_y3_d3
prebuy_i_cnt_y3_d7
prebuy_i_cnt_y3_d14
prebuy_i_cnt_y3_d30
prebuy_i_cnt_y3_d99
pref_orig_m3_1
pref_orig_m3_2
pref_orig_m3_3
pref_orig_m3_4
pref_orig_m3_5
pref_orig_m6_1
pref_orig_m6_2
pref_orig_m6_3
pref_orig_m6_4
pref_orig_m6_5
pref_orig_y1_1
pref_orig_y1_2
pref_orig_y1_3
pref_orig_y1_4
pref_orig_y1_5
pref_orig_y2_1
pref_orig_y2_2
pref_orig_y2_3
pref_orig_y2_4
pref_orig_y2_5
pref_orig_y3_1
pref_orig_y3_2
pref_orig_y3_3
pref_orig_y3_4
pref_orig_y3_5
pref_line_m3_1
pref_line_m3_2
pref_line_m3_3
pref_line_m3_4
pref_line_m3_5
pref_line_m6_1
pref_line_m6_2
pref_line_m6_3
pref_line_m6_4
pref_line_m6_5
pref_line_y1_1
pref_line_y1_2
pref_line_y1_3
pref_line_y1_4
pref_line_y1_5
pref_line_y2_1
pref_line_y2_2
pref_line_y2_3
pref_line_y2_4
pref_line_y2_5
pref_line_y3_1
pref_line_y3_2
pref_line_y3_3
pref_line_y3_4
pref_line_y3_5
pref_city_m3_1
pref_city_m3_2
pref_city_m3_3
pref_city_m3_4
pref_city_m3_5
pref_city_m6_1
pref_city_m6_2
pref_city_m6_3
pref_city_m6_4
pref_city_m6_5
pref_city_y1_1
pref_city_y1_2
pref_city_y1_3
pref_city_y1_4
pref_city_y1_5
pref_city_y2_1
pref_city_y2_2
pref_city_y2_3
pref_city_y2_4
pref_city_y2_5
pref_city_y3_1
pref_city_y3_2
pref_city_y3_3
pref_city_y3_4
pref_city_y3_5
pref_month_m3_1
pref_month_m3_2
pref_month_m3_3
pref_month_m3_4
pref_month_m3_5
pref_month_m6_1
pref_month_m6_2
pref_month_m6_3
pref_month_m6_4
pref_month_m6_5
pref_month_y1_1
pref_month_y1_2
pref_month_y1_3
pref_month_y1_4
pref_month_y1_5
pref_month_y2_1
pref_month_y2_2
pref_month_y2_3
pref_month_y2_4
pref_month_y2_5
pref_month_y3_1
pref_month_y3_2
pref_month_y3_3
pref_month_y3_4
pref_month_y3_5
flt_cnt_m3
flt_cnt_m6
flt_cnt_y1
flt_cnt_y2
flt_cnt_y3
next_flt_day
dist_cnt_y1
dist_cnt_y3
flt_nature_cnt_y1
flt_nature_cnt_y3
avg_dist_cnt_m3
avg_dist_cnt_m6
avg_dist_cnt_y1
avg_dist_cnt_y2
avg_dist_cnt_y3
complain_valid_cnt_m3
complain_valid_cnt_m6
complain_valid_cnt_y1
complain_valid_cnt_y2
complain_valid_cnt_y3
bag_cnt_m3
bag_cnt_m6
bag_cnt_y1
bag_cnt_y2
bag_cnt_y3
cabin_upgrd_cnt_m3
cabin_upgrd_cnt_m6
cabin_upgrd_cnt_y1
cabin_upgrd_cnt_y2
cabin_upgrd_cnt_y3
select_seat_cnt_m3
select_seat_cnt_m6
select_seat_cnt_y1
select_seat_cnt_y2
select_seat_cnt_y3
dist_d_cnt_m3
dist_d_cnt_m6
dist_d_cnt_y1
dist_d_cnt_y2
dist_d_cnt_y3
dist_i_cnt_m3
dist_i_cnt_m6
dist_i_cnt_y1
dist_i_cnt_y2
dist_i_cnt_y3
dist_all_cnt_m3
dist_all_cnt_m6
dist_all_cnt_y1
dist_all_cnt_y2
dist_all_cnt_y3
cabin_hd_cnt_m3
cabin_hd_cnt_m6
cabin_hd_cnt_y1
cabin_hd_cnt_y2
cabin_hd_cnt_y3
cabin_hi_cnt_m3
cabin_hi_cnt_m6
cabin_hi_cnt_y1
cabin_hi_cnt_y2
cabin_hi_cnt_y3
flt_leg_cnt_m3
flt_leg_cnt_m6
flt_leg_cnt_y1
flt_leg_cnt_y2
flt_leg_cnt_y3
flt_leg_i_cnt_m3
flt_leg_i_cnt_m6
flt_leg_i_cnt_y1
flt_leg_i_cnt_y2
flt_leg_i_cnt_y3
flt_leg_d_cnt_m3
flt_leg_d_cnt_m6
flt_leg_d_cnt_y1
flt_leg_d_cnt_y2
flt_leg_d_cnt_y3
flt_cancel_cnt_m3
flt_cancel_cnt_m6
flt_cancel_cnt_y1
flt_cancel_cnt_y2
flt_cancel_cnt_y3
flt_delay_cnt_m3
flt_delay_cnt_m6
flt_delay_cnt_y1
flt_delay_cnt_y2
flt_delay_cnt_y3
flt_delay_time_m3
flt_delay_time_m6
flt_delay_time_y1
flt_delay_time_y2
flt_delay_time_y3
pref_orig_city_m3
pref_orig_city_m6
pref_orig_city_y1
pref_orig_city_y2
pref_orig_city_y3
pref_dest_city_m3
pref_dest_city_m6
pref_dest_city_y1
pref_dest_city_y2
pref_dest_city_y3
cabin_fall_cnt_y3
flt_bag_cnt_m3
flt_bag_cnt_m6
flt_bag_cnt_y1
flt_bag_cnt_y2
flt_bag_cnt_y3
complain_cnt_m3
complain_cnt_m6
complain_cnt_y1
complain_cnt_y2
complain_cnt_y3
noshow_rate_m3
noshow_rate_m6
noshow_rate_y1
noshow_rate_y2
noshow_rate_y3
tkt_3y_amt
tkt_d_amt_m3
tkt_d_amt_m6
tkt_d_amt_y1
tkt_d_amt_y2
tkt_d_amt_y3
tkt_i_amt_m3
tkt_i_amt_m6
tkt_i_amt_y1
tkt_i_amt_y2
tkt_i_amt_y3
tkt_all_amt_m3
tkt_all_amt_m6
tkt_all_amt_y1
tkt_all_amt_y2
tkt_all_amt_y3
tkt_avg_amt_m3
tkt_avg_amt_m6
tkt_avg_amt_y1
tkt_avg_amt_y2
tkt_avg_amt_y3
tkt_return_cnt_m3
tkt_return_cnt_m6
tkt_return_cnt_y1
tkt_return_cnt_y2
tkt_return_cnt_y3
tkt_book_cnt_m3
tkt_book_cnt_m6
tkt_book_cnt_y1
tkt_book_cnt_y2
tkt_book_cnt_y3
tkt_return_all_cnt_m3
tkt_return_all_cnt_m6
tkt_return_all_cnt_y1
tkt_return_all_cnt_y2
tkt_return_all_cnt_y3
tkt_avg_interval_m3
tkt_avg_interval_m6
tkt_avg_interval_y1
tkt_avg_interval_y2
tkt_avg_interval_y3
pit_all_amt
pit_out_amt
pit_accu_non_cnt
pit_accu_air_cnt
pit_accu_air_amt
pit_accu_non_amt
pit_now_cons_amt
pit_next_level_dist
pit_next_level_leg
mdl_mcv
mdl_lost_idx
mdl_influence
pit_accu_amt_m3
pit_accu_amt_m6
pit_accu_amt_y1
pit_accu_amt_y2
pit_accu_amt_y3
pit_cons_amt_m3
pit_cons_amt_m6
pit_cons_amt_y1
pit_cons_amt_y2
pit_cons_amt_y3
pit_avg_accu_amt_m3
pit_avg_accu_amt_m6
pit_avg_accu_amt_y1
pit_avg_accu_amt_y2
pit_avg_accu_amt_y3
pit_avg_cons_amt_m3
pit_avg_cons_amt_m6
pit_avg_cons_amt_y1
pit_avg_cons_amt_y2
pit_avg_cons_amt_y3
pit_cons_cnt_m3
pit_cons_cnt_m6
pit_cons_cnt_y1
pit_cons_cnt_y2
pit_cons_cnt_y3
pit_add_air_amt_m3
pit_add_air_amt_m6
pit_add_air_amt_y1
pit_add_air_amt_y2
pit_add_air_amt_y3
pit_des_upg_amt_m3
pit_des_upg_amt_m6
pit_des_upg_amt_y1
pit_des_upg_amt_y2
pit_des_upg_amt_y3
pit_add_air_cnt_y2
pit_add_air_cnt_y3
pit_add_non_cnt_m3
pit_add_non_cnt_m6
pit_add_non_cnt_y1
pit_add_non_cnt_y2
pit_add_non_cnt_y3
pit_add_buy_cnt_m3
pit_add_buy_cnt_m6
pit_add_buy_cnt_y1
pit_add_buy_cnt_y2
pit_add_buy_cnt_y3
pit_add_mnl_cnt_m3
pit_add_mnl_cnt_m6
pit_add_mnl_cnt_y1
pit_add_mnl_cnt_y2
pit_add_mnl_cnt_y3
pit_add_ech_cnt_m3
pit_add_ech_cnt_m6
pit_add_ech_cnt_y1
pit_add_ech_cnt_y2
pit_add_ech_cnt_y3
pit_add_oth_cnt_m3
pit_add_oth_cnt_m6
pit_add_oth_cnt_y1
pit_add_oth_cnt_y2
pit_add_oth_cnt_y3
pit_des_tkt_cnt_m3
pit_des_tkt_cnt_m6
pit_des_tkt_cnt_y1
pit_des_tkt_cnt_y2
pit_des_tkt_cnt_y3
pit_des_bag_cnt_m3
pit_des_bag_cnt_m6
pit_des_bag_cnt_y1
pit_des_bag_cnt_y2
pit_des_bag_cnt_y3
pit_des_upg_cnt_m3
pit_des_upg_cnt_m6
pit_des_upg_cnt_y1
pit_des_upg_cnt_y2
pit_des_upg_cnt_y3
pit_des_mall_cnt_m3
pit_des_mall_cnt_m6
pit_des_mall_cnt_y1
pit_des_mall_cnt_y2
pit_des_mall_cnt_y3
pit_des_selt_cnt_m3
pit_des_selt_cnt_m6
pit_des_selt_cnt_y1
pit_des_selt_cnt_y2
pit_des_selt_cnt_y3
pit_des_out_cnt_m3
pit_des_out_cnt_m6
pit_des_out_cnt_y1
pit_des_out_cnt_y2
pit_des_out_cnt_y3
pit_des_oth_cnt_m3
pit_des_oth_cnt_m6
pit_des_oth_cnt_y1
pit_des_oth_cnt_y2
pit_des_oth_cnt_y3
pit_avg_amt_m3
pit_avg_amt_m6
pit_avg_amt_y1
pit_avg_amt_y2
pit_avg_amt_y3
pit_avg_interval_m3
pit_avg_interval_m6
pit_avg_interval_y1
pit_avg_interval_y2
pit_avg_interval_y3
pit_ech_avg_amt_m3
pit_ech_avg_amt_m6
pit_ech_avg_amt_y1
pit_ech_avg_amt_y2
pit_ech_avg_amt_y3
pit_out_avg_amt_m3
pit_out_avg_amt_m6
pit_out_avg_amt_y1
pit_out_avg_amt_y2
pit_out_avg_amt_y3
pit_income_cnt_m3
pit_income_cnt_m6
pit_income_cnt_y1
pit_income_cnt_y2
pit_income_cnt_y3
pit_pay_cnt_m3
pit_pay_cnt_m6
pit_pay_cnt_y1
pit_pay_cnt_y2
pit_pay_cnt_y3
pit_income_avg_amt_m3
pit_income_avg_amt_m6
pit_income_avg_amt_y1
pit_income_avg_amt_y2
pit_income_avg_amt_y3
pit_pay_avg_amt_m3
pit_pay_avg_amt_m6
pit_pay_avg_amt_y1
pit_pay_avg_amt_y2
pit_pay_avg_amt_y3"""


str2 = """pax_fcny
pax_tax
dist_cnt_y1
dist_cnt_y3
dist_i_cnt_y2
dist_i_cnt_y3
dist_all_cnt_y2
avg_dist_cnt_y1
avg_dist_cnt_y3
avg_dist_cnt_y2
pref_city_y2_2
pref_city_y3_2
pref_city_y3_3
pref_city_y2_3
pref_city_y3_4
pref_month_y2_3
pref_month_y3_2
pref_month_y3_1
pref_month_y2_1
pref_month_y3_3
pref_month_y3_5
pref_month_y2_2
ffp_nbr
seg_route_to
age
tkt_3y_amt
seg_cabin
seat_walkway_cnt_y3
seat_middle_cnt_y3
seat_window_cnt_y3
select_seat_cnt_y2
select_seat_cnt_y3
flt_bag_cnt_y3
pref_line_y2_2
pref_line_y3_1
pref_line_y3_4
pref_line_y3_2
pref_line_y3_3
flt_delay_cnt_y3
flt_bag_cnt_y2
tkt_avg_amt_y2
tkt_avg_amt_y3
tkt_all_amt_y3
tkt_i_amt_y2
tkt_i_amt_y3
tkt_avg_interval_y2
pit_accu_amt_y1
pit_accu_air_amt
pit_avg_amt_y2
pit_avg_amt_y3
pref_aircraft_y3_1
pref_aircraft_y3_2
pref_aircraft_y3_3
pref_aircraft_y3_4"""

str1 = """pax_fcny	pax_tax	tkt_3y_amt	dist_cnt_y1	avg_dist_cnt_y2	tkt_avg_amt_y2	tkt_avg_amt_y3	seg_cabin	pref_city_y3_3	dist_i_cnt_y3	pref_month_y3_2	dist_cnt_y3	tkt_i_amt_y3	avg_dist_cnt_y3	pref_city_y2_2	avg_dist_cnt_y1	pref_month_y2_3	ffp_nbr	seg_route_to	pref_month_y3_1	age	dist_i_cnt_y2	seat_walkway_cnt_y3	pref_city_y2_3	flt_bag_cnt_y3	pref_month_y2_1	pref_city_y3_2	select_seat_cnt_y2	pref_line_y2_2	dist_all_cnt_y2	seat_window_cnt_y3	flt_delay_cnt_y3	seat_middle_cnt_y3	pref_line_y3_1	pref_month_y3_3	pit_avg_amt_y3	pref_line_y3_4	pref_city_y3_4	tkt_all_amt_y3	pref_month_y2_2	select_seat_cnt_y3	pref_line_y3_3	flt_bag_cnt_y2	pit_accu_amt_y1	tkt_i_amt_y2	pref_line_y3_2	pref_month_y3_5	tkt_avg_interval_y2	pit_avg_amt_y2	pit_accu_air_amt	emd_lable2	emd_lable	pax_name	pax_passport"""

List=str.split('\n')
# print(list)

# list2 = [2,3,318,208,209,215,338,10,0,216,246,245,172,174,337,250,307,168,329,198,56,197,150,169,5,235,51,4,333,306,147,296,148,101,149,45,194,472,199,142,214,123,193,41,363,287,192,1,327,201]
# for i in list2:
#     print(list[i])

feature1 = train_data[List]
# # feature2 = test_data[list]
# #
# feature1.to_csv('./dataset/train_class_005.csv',index=False,encoding='utf_8_sig')
#将文本标签转化成数字
# ordinal_encoder = OrdinalEncoder()
#
# text_encoded = ordinal_encoder.fit_transform(feature1)
# # text_encoded2 = ordinal_encoder.fit_transform(feature2)
#
# text_encoded[:10]
# # text_encoded2[:10]
# ordinal_encoder.categories_  #查看一下映射
#
# train_data[List] = text_encoded
# # test_data[list] = text_encoded2

train_data.head()
# test_data.head()

# train_data.to_csv('./dataset/data_sum_02.csv',index=False,encoding='utf_8_sig')

max_feature = 150
features_columns = [col for col in List if col not in ['pax_name', 'pax_passport', 'emd_lable2','emd_lable']]    #直接使用list读取数据

#features_columns_test = [col for col in test_data.columns ]

train = train_data[features_columns].values
target =train_data['emd_lable'].values

feature1 = train_data[features_columns]

#
#
# from sklearn.decomposition import PCA
#
# pca_sk = PCA(n_components=80)
# newMat = pca_sk.fit_transform(feature1)
# target = target.reshape((23432,-1))
# newMat = np.hstack((newMat,target))
# data1 = DataFrame(newMat)
# # data1 = newMat.append(train_data['emd_lable2'])
# data1.to_csv('./dataset/train_005.csv',index=False,encoding='utf_8_sig')
#

from sklearn.model_selection import train_test_split
import lightgbm
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

#print(test.shape, target_test.shape)

clf = lightgbm

train_matrix = clf.Dataset(train, label=target)
test_matrix = clf.Dataset(train, label=target)
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

# from sklearn.feature_selection import SelectFromModel
# from sklearn.ensemble import GradientBoostingClassifier
#
# #GBDT作为基模型的特征选择
# Klist = SelectFromModel(GradientBoostingClassifier()).fit_transform(train, target)
# print(Klist)


def combine(outputList, sortList):
    CombineList = list()
    for index in range(len(outputList)):
        CombineList.append((outputList[index], sortList[index]))
    return CombineList


SList = combine(List, FIList)
SList.sort(key=lambda x: x[1], reverse=True)
IndexList = []
for i in range(max_feature):
    IndexList.append(SList[i][0])

print(IndexList)
IndexList.append("emd_lable2")
IndexList.append("emd_lable")
IndexList.append("pax_name")
IndexList.append("pax_passport")

Data = train_data[IndexList]
Data.to_csv('./dataset/train_twoStep_501.csv',index=False,encoding='utf_8_sig')

# test_data.to_csv('val_data02.csv',index=False,encoding='utf_8_sig')
# DF = pd.DataFrame(train_data)
#
# def ifDel(i):
#     x = 0
#     x += 1 if DF.loc[i,'age'] else 0
#     x += 1 if DF.loc[i,'avg_dist_cnt_y3'] else 0
#     x += 1 if DF.loc[i,'tkt_avg_amt_y3'] else 0
#     x += 1 if DF.loc[i,'pax_fcny'] else 0
#     if(x>1): return 1
#     return 0
#
# for i in range(DF.shape[0]):
#     if(ifDel(i) == 0):
#         DF.drop([i])
# train_data.to_csv('./dataset/train5.csv')