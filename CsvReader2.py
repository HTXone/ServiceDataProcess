import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

data = pd.read_csv('./dataset/train_data.csv',encoding='utf-8')
# data2 = pd.read_csv('./dataset/val_data.csv',encoding='utf-8')
print(data.shape)
print(data2.shape)
#data.head()

str = """pax_name
pax_passport
pax_fcny
pax_tax
emd_lable
emd_lable2
tkt_3y_amt
dist_cnt_y1
dist_cnt_y3
avg_dist_cnt_y2
tkt_avg_amt_y3
ffp_nbr
seg_route_to
avg_dist_cnt_y3
dist_i_cnt_y3
dist_i_cnt_y2
pref_city_y3_1
pref_city_y3_3
tkt_avg_amt_y2
dist_all_cnt_y2
flt_bag_cnt_y3
pref_city_y2_2
tkt_all_amt_m3
pref_month_y3_2
prebuy_d_cnt_m3_d99
pref_month_y3_1
pref_line_y3_4
pref_city_y2_3
age
select_seat_cnt_y2
seat_middle_cnt_y3
gender
tkt_all_amt_y3
flt_bag_cnt_y2
pref_line_y3_1
pref_orig_city_y3
pref_line_y3_2
prebuy_i_cnt_y3_d99
pref_line_y3_3
seat_walkway_cnt_y2
pref_month_y2_3
pit_avg_amt_y3
pref_month_y3_3
pref_line_y2_1
avg_dist_cnt_y1
pref_orig_y3_2
pref_month_y2_2
seat_window_cnt_y3
pit_accu_air_amt
flt_delay_time_m3
pref_month_y2_1
seg_cabin
tkt_i_amt_y2
pref_month_y3_5"""

""""pax_fcny	pax_tax tkt_3y_amt	dist_cnt_y1	dist_cnt_y3	avg_dist_cnt_y2	tkt_avg_amt_y3	ffp_nbr avg_dist_cnt_y3	dist_i_cnt_y3	dist_i_cnt_y2
""""

list=str.split('\n')
print(list)

train_data = data[list]
print(train_data.shape)
val_data = data2[list]
print(val_data.shape)

train_data.to_csv('train_data01.csv',index=False,encoding='utf_8_sig')
val_data.to_csv('val_data01.csv',index=False,encoding='utf_8_sig')