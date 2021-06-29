import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series,DataFrame

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import OrdinalEncoder

train_data_file = './dataset/trainFinal.csv'
# test_data_file = './val_data01.csv'
train_data = pd.read_csv(train_data_file, encoding='utf-8')
# test_data = pd.read_csv(test_data_file, encoding='utf-8')

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', 10)
#train_data.head(50)
print(train_data.shape)
# print(test_data.shape)

train_data["seg_cabin"].fillna('Y', inplace=True) #填充

Names = """pax_fcny	pax_tax	tkt_3y_amt	dist_cnt_y1	dist_cnt_y3	avg_dist_cnt_y2	tkt_avg_amt_y3	dist_i_cnt_y3	avg_dist_cnt_y3	pref_city_y3_3	tkt_avg_amt_y2	tkt_i_amt_y3	pref_city_y3_2	dist_i_cnt_y2	seat_walkway_cnt_y3	dist_all_cnt_y2	pref_month_y3_1	pref_city_y2_2	pref_line_y3_3	pref_city_y2_3	tkt_all_amt_y3	avg_dist_cnt_y1	pref_month_y3_2	pref_line_y2_2	pref_line_y3_1	prebuy_i_cnt_y3_d99	pref_line_y3_4	pref_line_y3_2	seg_route_to	ffp_nbr	pref_orig_y3_3	flt_bag_cnt_y3	tkt_i_amt_y2	seat_middle_cnt_y3	pref_line_y2_1	seat_window_cnt_y3	pref_orig_city_y2	pref_month_y2_1	pref_line_y2_3	pref_line_y2_4	pref_month_y2_2	pref_month_y3_3	pref_orig_y3_2	tkt_all_amt_y2	age	flt_delay_time_m3	flt_delay_cnt_y3	pref_city_y1_2	pref_city_y3_4	prebuy_i_cnt_y2_d99	pit_avg_amt_y3	seat_walkway_cnt_y2	nation_name	pref_orig_y2_2	pref_month_y2_3	flt_bag_cnt_y2	pref_month_y3_4	pit_all_amt	pit_avg_amt_y2	pit_accu_air_amt	cabin_y_cnt_y3	cabin_hy_cnt_y3	pref_line_y1_2	pit_next_level_dist	flt_nature_cnt_y1	pref_city_y2_4	dist_all_cnt_y1	pref_orig_y2_3	dist_d_cnt_y3	pit_avg_accu_amt_y3	tkt_avg_amt_y1	flt_delay_time_y2	pit_avg_interval_y2	pref_city_y1_1	pref_line_y3_5	mdl_mcv	dist_i_cnt_y1	dist_all_cnt_y3	pit_accu_air_cnt	gender	tkt_book_cnt_y3	mdl_influence	flt_nature_cnt_y3	seat_middle_cnt_y2	select_seat_cnt_y2	flt_leg_i_cnt_y3	avg_dist_cnt_m6	tkt_d_amt_y3	pref_city_y1_3	pref_month_y3_5	cabin_y_cnt_y2	pref_orig_city_y3	pit_accu_amt_y3	prebuy_i_cnt_y3_d30	prebuy_i_cnt_y3_d3	flt_delay_time_m6	pref_month_y2_4	prebuy_d_cnt_y3_d99	pit_accu_amt_y2	pit_avg_interval_y3	seg_cabin	tkt_avg_interval_y3	flt_delay_time_y3	pref_month_y1_1	dist_i_cnt_m6	pref_line_y1_3	pit_add_air_amt_y3	pref_month_y2_5	flt_leg_cnt_y2	pit_accu_amt_y1	flt_delay_cnt_y2	pit_avg_accu_amt_y2	pref_city_y3_1	dist_all_cnt_m6	seat_window_cnt_y2	flt_cnt_y3	pref_line_y1_1	pref_orig_y3_1	flt_bag_cnt_y1	flt_leg_i_cnt_y2	prebuy_d_cnt_y2_d99	tkt_d_amt_y2	flt_delay_time_y1	tkt_avg_interval_y2	dist_d_cnt_y2	pref_orig_y1_1	cabin_hy_cnt_y2	pref_orig_city_y1	cabin_hf_cnt_m6	pref_orig_y3_4	pref_dest_city_y3	pit_accu_non_amt	tkt_book_cnt_y2	tkt_i_amt_y1	pref_city_y2_5	tkt_all_amt_y1	pref_city_y2_1	residence_country	pref_orig_y2_4	pit_add_air_amt_y2	prebuy_i_cnt_y3_d14	pref_orig_y1_2	tkt_return_cnt_y3	pit_avg_accu_amt_y1	pref_line_y2_5	pit_cons_amt_y3	prebuy_i_cnt_y2_d30	pref_orig_y3_5	pit_avg_amt_y1	pref_orig_city_m6	emd_lable2	emd_lable	pax_name	pax_passport"""

List=Names.split('\t')

train_data.head()

max_feature = 150
features_columns = [col for col in List if col not in ['pax_name', 'pax_passport', 'emd_lable2','emd_lable']]    #直接使用list读取数据

feature1 = train_data[List]

feature2 = feature1[features_columns]

# feature1.to_csv('./dataset/train_class_005.csv',index=False,encoding='utf_8_sig')
# 将文本标签转化成数字
ordinal_encoder = OrdinalEncoder()

text_encoded = ordinal_encoder.fit_transform(feature2)
# text_encoded2 = ordinal_encoder.fit_transform(feature2)

text_encoded[:10]
# text_encoded2[:10]
ordinal_encoder.categories_  #查看一下映射

feature1[features_columns] = text_encoded
# test_data[list] = text_encoded2

# train_data.head()
# test_data.head()

feature1.to_csv('./dataset/trainFinal2.csv',index=False,encoding='utf_8_sig')

# train = train_data[features_columns].values
# target =train_data['emd_lable'].values



feature1 = train_data[features_columns]

