import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series,DataFrame

from mpl_toolkits.mplot3d import Axes3D

#显示所有列
pd.set_option('display.max_columns', None)

#显示所有行
pd.set_option('display.max_rows', None)

#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

# Train_data = pd.read_csv('./dataset/train_data01.csv')
# print(Train_data.shape)
# sum = 0
# SumList = [0,0,0]
# PreList = [0,0,0]
# PreList2 = [0,0]
# SeatList = [0,0,0]
# SeatList2 = [0,0,0]
# for i in Train_data.iterrows():
#     # print(i[1]['seat_middle_cnt_y3'],i[1]['seat_window_cnt_y3'],i[1]['seat_walkway_cnt_y3'])
#     k = i[1]['seat_middle_cnt_y3']+i[1]['seat_window_cnt_y3']+i[1]['seat_walkway_cnt_y3']
#     if(k>0):
#         sum+=1
#         if(k<=3):
#             SumList[0]+=1
#             if(i[1]["emd_lable2"]>0):
#                 PreList[2]+=1
#         else:
#             SumList[1]+=1
#
#             if(i[1]['seat_middle_cnt_y3'] / k > 0.6):
#                 SeatList2[0]+=1
#                 if (i[1]["emd_lable2"] > 0):
#                     SeatList[0] += 1
#             if (i[1]['seat_window_cnt_y3'] / k > 0.6):
#                 SeatList2[1] += 1
#                 if (i[1]["emd_lable2"] > 0):
#                     SeatList[1] += 1
#             if (i[1]['seat_walkway_cnt_y3'] / k > 0.6):
#                 SeatList2[2] += 1
#                 if (i[1]["emd_lable2"] > 0):
#                     SeatList[2] += 1
#
# print(sum,SumList,PreList,PreList2,SeatList,SeatList2)

def leg(x):
    if x==0:
        return 0
    elif x>0 and x <=5:
        return 1
    elif x > 5 and x <= 10:
        return 2
    elif x > 10 and x <= 15:
        return 3
    elif x > 15 and x <= 20:
        return 4
    elif x > 20 and x <= 25:
        return 5
    elif x > 25 and x <= 30:
        return 6
    elif x > 30 and x <= 35:
        return 7
    elif x > 35:
        return 8


Train_data = pd.read_csv('./dataset/train_data02.csv')
print(Train_data.shape)
SUM = np.zeros((9,9))
SUM2 = np.zeros((9,9))
SUM3 = np.zeros((9,9))
for i in Train_data.iterrows():
    # print(i[1]['residence_country'],i[1]['nation_name'])
    # SUM[(int)(i[1]['residence_country'])][int(i[1]['nation_name'])]+=1
    # if(i[1]["emd_lable2"] > 0):
    #     SUM2[(int)(i[1]['residence_country'])][int(i[1]['nation_name'])] += 1
    SUM[leg((int)(i[1]['flt_leg_i_cnt_y3']))][leg((int)(i[1]['flt_leg_d_cnt_y3']))]+=1
    if(i[1]["emd_lable2"] > 0):
        SUM2[leg((int)(i[1]['flt_leg_i_cnt_y3']))][leg((int)(i[1]['flt_leg_d_cnt_y3']))] += 1

X = []
Y = []
Z = []
for i in range(0,9):
    for j in range(0,9):
        SUM3[i][j] = SUM2[i][j]/SUM[i][j]  if(SUM[i][j]>0) else 0
        X.append(i)
        Y.append(j)
        Z.append(SUM3[i][j])

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(xs=X,ys=Y,zs=Z)  # 绘图
plt.show()



