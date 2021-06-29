import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series,DataFrame

from mpl_toolkits.mplot3d import Axes3D



class Classify:
    def __init__(self):
        self.List = np.zeros((9))
        self.Ans = []

    def CList(self):
        self.classList = {}
        if(self.List[0]):
            self.classList['age'] = np.zeros(7)
        if (self.List[1]):
            self.classList['mem_lev'] = np.zeros(5)
        if (self.List[2]):
            self.classList['flt_cnt'] = np.zeros(6)
        if (self.List[3]):
            self.classList['mdl_mcv'] = np.zeros(6)
        if (self.List[4]):
            self.classList['avg_dist'] = np.zeros(8)
        if (self.List[5]):
            self.classList['tkt_avg'] = np.zeros(9)
        if (self.List[6]):
            self.classList['rout_to'] = np.zeros(5)
        if (self.List[7]):
            self.classList['pref_aircarft'] = np.zeros(7)
        if (self.List[8]):
            self.classList['gender'] = np.zeros(3)
        if (self.List[9]):
            self.classList['cabin'] = np.zeros(6)
        if (self.List[10]):
            self.classList['leg_cnt'] = np.zeros(5)
        if (self.List[11]):
            self.classList['seat'] = np.zeros(3)

        # print(self.classList)

    def Clist2(self):
        self.Klist = 1
        if (self.List[0]):
            self.Klist *= 6
        if (self.List[1]):
            self.Klist *= 6
        if (self.List[2]):
            self.Klist *= 3
        if (self.List[3]):
            self.Klist *= 6
        if (self.List[4]):
            self.Klist *= 5
        if (self.List[5]):
            self.Klist *= 3
        if (self.List[6]):
            self.Klist *= 8
        if (self.List[7]):
            self.Klist *= 5
        if (self.List[8]):
            self.Klist *= 5

        # print(self.Klist)
        self.classList = np.zeros(self.Klist)
        self.classList2 = np.zeros(self.Klist)

    def AddSum(self,i):
        t = 1
        T = 0
        if (self.List[0]):
            T += self.Klist/(6*t) * self.flt_cnt(i[1]['flt_cnt_y3'])
            t *= 6
        if (self.List[1]):
            T += self.Klist/(6*t) * self.Mdl_mcv(i[1]['mdl_mcv'])
            t *= 6
        if (self.List[2]):
            T += self.Klist / (3 * t) * self.Gender(i[1]['gender'])
            t *= 3
        if (self.List[3]):
            T += self.Klist / (6 * t) * self.Cabin(i[1]['cabin'])
            t *= 6
        if (self.List[4]):
            T += self.Klist / (5 * t) * self.leg_cnt(i[1]['flt_leg_i_cnt_y3'], i[1]['flt_leg_d_cnt_y3'])
            t *= 5
        if (self.List[5]):
            T += self.Klist / (3 * t) * self.Seat(i[1]['seat'])
            t *= 3
        if (self.List[6]):
            T += self.Klist / (8 * t) * self.Pax(i[1]['pax_fcny'])
            t *= 8
        if (self.List[7]):
            T += self.Klist / (5 * t) * self.set_cabin(i[1]['seg_cabin'])
            t *= 5
        if (self.List[8]):
            T += self.Klist / (5 * t) * self.country(i[1]['country1'],i[1]['country2'])
            t *= 5
        self.classList[(int)(T)]+=1
        if(i[1]['emd_lable2']>0):
            self.classList2[(int)(T)]+=1

        return T

    def Cal(self,List,len,sum):         #计算属性百分比
        K = np.zeros(len)
        for i in range(len):
            K[i] = List[i]/sum
        return K

    def Cal2(self,x,y):                 #计算增长趋势
        if (x>0):
            return (y-x)/x
        else:
            return 0



    # def Age(self,x):
    #     if(x == '0' or x == 0):
    #         return 0
    #     elif(x=='0-10'):
    #         return 1
    #     elif (x =='21-30'):
    #         return 2
    #     elif (x =='31-40'):
    #         return 3
    #     elif (x =='41-50'):
    #         return 4
    #     elif (x =='51-60'):
    #         return 5
    #     elif (x == '60+'):
    #         return 6
    #
    # def MemberLev(self,x):
    #     if(x == 0 or x =='0'):
    #         return 0
    #     elif(x == 'SILHL'):
    #         return 1
    #     elif(x == 'STD'):
    #         return 2
    #     elif(x == 'GOLHL'):
    #         return 3
    #     else:
    #         return 4

    # def flt_cnt(self,x):
    #     if(x == 0):
    #         return 0
    #     elif(x>0 and x<=5):
    #         return 1
    #     elif(x>5 and x<=10):
    #         return 2
    #     elif(x>10 and x<=15):
    #         return 3
    #     elif(x>15 and x<=20):
    #         return 4
    #     else:
    #         return 5

    # def Mdl_mcv(self,x):
    #     if(x == 0):
    #         return 0
    #     elif(x>0 and x<=50):
    #         return 1
    #     elif(x>50 and x<=100):
    #         return 2
    #     elif(x>100 and x<=150):
    #         return 3
    #     elif(x>150 and x<=200):
    #         return 4
    #     else:
    #         return 5

    # def avg_dist(self,x):
    #     if(x == 0):
    #         return 0
    #     elif(x>0 and x<=2000):
    #         return 1
    #     elif(x>2000 and x<=4000):
    #         return 2
    #     elif(x>4000 and x<=6000):
    #         return 3
    #     elif(x>6000 and x<=8000):
    #         return 4
    #     elif(x>8000 and x<=10000):
    #         return 5
    #     elif(x>10000 and x<=12000):
    #         return 6
    #     else:
    #         return 7

    def tkt_avg_amt(self,x):
        if(x == 0):
            return 0
        elif(x >0 and x<=10000):
            return 1
        elif (x > 10000 and x <= 20000):
            return 2
        elif (x > 20000 and x <= 30000):
            return 3
        elif (x > 30000 and x <= 40000):
            return 4
        elif (x > 40000 and x <= 60000):
            return 5
        elif (x > 60000 and x <= 80000):
            return 6
        elif (x > 80000 and x <= 100000):
            return 7
        else:
            return 8

    def tkt_avg_amt2(self,x):
        if(x == 0):
            return 0
        elif(x >0 and x<=1000):
            return 1
        elif (x > 1000 and x <= 2000):
            return 2
        elif (x > 2000 and x <= 3000):
            return 3
        elif (x > 3000 and x <= 4000):
            return 4
        elif (x > 4000 and x <= 6000):
            return 5
        elif (x > 6000 and x <= 8000):
            return 6
        elif (x > 8000 and x <= 10000):
            return 7
        else:
            return 8

    # def Seg_rout_to(self,x):
    #     if(x == 0):
    #         return 0
    #     elif(x == 'JFK'):
    #         return 1
    #     elif(x=='LAX'):
    #         return 2
    #     elif(x=='SYD'):
    #         return 3
    #     else:
    #         return 4

    # def pref_aircraft(self,x):
    #     return 1

    # def Gender(self,x):
    #     return x
    #     # if(x == 0 or x == '0'or x == 'U'):
    #     #     return 0
    #     # elif(x == 'M'):
    #     #     return 1
    #     # elif(x=='F'):
    #     #     return 2

    # def Cabin(self,x1,x2,x3,x4,x5):
    #     # return x
    #     x = max(x1,x2,x3,x4,x5)
    #     if(x == 0):
    #         return 0
    #     elif(x == x1):
    #         return 1
    #     elif (x == x2):
    #         return 2
    #     elif (x == x3):
    #         return 3
    #     elif (x == x4):
    #         return 4
    #     elif (x == x5):
    #         return 5

    def avg_dist_cnt(self,x):
        if (x == 0):
            return 0
        elif (x > 0 and x <= 2000):
            return 1
        elif (x > 2000 and x <= 4000):
            return 2
        elif (x > 4000 and x <= 6000):
            return 3
        elif (x > 6000 and x <= 8000):
            return 4
        else:
            return 5

    def ffp_nbr(self,x):
        if (x == 0):
            return 0
        elif (x > 0 and x <= 610000000000):
            return 1
        elif (x > 610000000000 and x <= 620000000000):
            return 2
        elif (x > 620000000000 and x <= 630000000000):
            return 3
        elif (x > 630000000000 and x <= 640000000000):
            return 4
        elif (x > 640000000000 and x <= 650000000000):
            return 5
        else:
            return 6

    def tkt_3y(self,x):
        if (x == 0):
            return 0
        elif (x > 0 and x <= 2000):
            return 1
        elif (x > 2000 and x <= 4000):
            return 2
        elif (x > 4000 and x <= 6000):
            return 3
        elif (x > 6000 and x <= 8000):
            return 4
        elif (x > 8000 and x <= 10000):
            return 5
        elif (x > 10000 and x <= 12000):
            return 6
        else:
            return 7

    def dist_cnt(self,x):
        if(x==0):
            return 0
        elif(x>0 and x<=10000):
            return 1
        elif(x>10000 and x<=20000):
            return 2
        elif(x>20000 and x<=30000):
            return 3
        elif(x>30000 and x<=40000):
            return 4
        elif (x > 40000 and x <= 50000):
            return 5
        else:
            return 6

    def dist_i_cnt(self,x):
        if (x == 0):
            return 0
        elif (x > 0 and x <= 10000):
            return 1
        elif (x > 10000 and x <= 20000):
            return 2
        elif (x > 20000 and x <= 30000):
            return 3
        elif (x > 30000 and x <= 40000):
            return 4
        elif (x > 40000 and x <= 50000):
            return 5
        else:
            return 6

    def dist_all_cnt(self,x):
        if (x == 0):
            return 0
        elif (x > 0 and x <= 100000):
            return 1
        elif (x > 100000 and x <= 200000):
            return 2
        elif (x > 200000 and x <= 300000):
            return 3
        else:
            return 4

    def Seat(self,x,y,z):
        # return x
        s = x+y+z
        if(s == 0):
            return 0
        s = max(x/s,y/s,z/s)
        if(s >= 0.6):
            return 1
        else:
            return 2

    def Pax_tax(self,x):
        if(x==0):
            return 0
        if(x>0 and x<=1000):
            return 1
        if (x > 1000 and x <= 2000):
            return 2
        if (x > 2000 and x <= 3000):
            return 3
        if (x > 3000 and x <= 4000):
            return 4
        if (x > 4000 and x <= 5000):
            return 5
        if (x > 5000 and x <= 6000):
            return 6
        if (x > 6000):
            return 7

    def Pax_fcny(self,x):
        if (x == 0):
            return 0
        if (x > 0 and x <= 1000):
            return 1
        if (x > 1000 and x <= 2000):
            return 2
        if (x > 2000 and x <= 3000):
            return 3
        if (x > 3000 and x <= 4000):
            return 4
        if (x > 4000 and x <= 5000):
            return 5
        if (x > 5000 and x <= 6000):
            return 6
        if (x > 6000):
            return 7

    def select_seat(self,x):
        if (x == 0):
            return 0
        elif (x > 0 and x <= 2):
            return 1
        elif (x > 2 and x <= 4):
            return 2
        elif (x > 4 and x <= 6):
            return 3
        else:
            return 4

    def flt_bag(self,x):
        if (x == 0):
            return 0
        elif (x > 0 and x <= 20):
            return 1
        elif (x > 20 and x <= 40):
            return 2
        elif (x > 40 and x <= 60):
            return 3
        elif (x > 60 and x <= 80):
            return 4
        else:
            return 5

    def flt_delay(self,x):
        if (x == 0):
            return 0
        elif (x > 0 and x <= 10):
            return 1
        elif (x > 10 and x <= 20):
            return 2
        elif (x > 20 and x <= 30):
            return 3
        else:
            return 4

    def pref_aircraft(self,x):
        if(x == 0 or x =="0" or x==""):
            return 0
        if(type(x).__name__ == "str"):
            if(x[0] == "3"):
                return 1
            elif(x[0] == "7"):
                return 2
        if(x>400):
            return 2
        else:
            return 1

    def pit_accu_amt(self,x):
        if (x == 0):
            return 0
        elif (x > 0 and x <= 2000):
            return 1
        elif (x > 2000 and x <= 4000):
            return 2
        elif (x > 4000 and x <= 6000):
            return 3
        elif (x > 6000 and x <= 8000):
            return 4
        elif (x > 8000 and x <= 10000):
            return 5
        elif (x > 10000 and x <= 12000):
            return 6
        else:
            return 7

    def pit_accu_air_amt(self,x):
        if (x == 0):
            return 0
        elif (x > 0 and x <= 2000):
            return 1
        elif (x > 2000 and x <= 4000):
            return 2
        elif (x > 4000 and x <= 6000):
            return 3
        elif (x > 6000 and x <= 8000):
            return 4
        elif (x > 8000 and x <= 10000):
            return 5
        elif (x > 10000 and x <= 12000):
            return 6
        else:
            return 7

    def pit_avg_amt(self,x):
        if (x == 0):
            return 0
        elif (x > 0 and x <= 2000):
            return 1
        elif (x > 2000 and x <= 4000):
            return 2
        elif (x > 4000 and x <= 6000):
            return 3
        elif (x > 6000 and x <= 8000):
            return 4
        elif (x > 8000 and x <= 10000):
            return 5
        elif (x > 10000 and x <= 12000):
            return 6
        else:
            return 7

    def country(self,x):
        if (x == 0 or x == '0'):
            return 0
        if (x in ['澳大利亚', '法国', '美国', '日本', '新加坡', '意大利', '英国', '爱尔兰', '奥地利', '百慕大', '比利时', '波多黎各', '波兰', '大不列颠联合王国',
                  '德国', '俄罗斯', '芬兰', '韩国', '荷兰', '加拿大', '捷克', '挪威', '瑞典', '西班牙', '新西兰', '以色列']):
            return 1
        return 2

    def country2(self,x,y):
        if (x==0 or y ==0):
            return 0
        if(x == 1 and y ==1):
            return 1
        if(x == 2 and y ==2):
            return 2
        if(x ==1 and y ==2):
            return 3
        if(x ==2 and y ==1):
            return 4

    def Visit(self):
        for i in range(0,self.Klist):
            if(self.classList[i]>30):
                z = self.classList2[i]/self.classList[i]
                if(z>0.2):
                    print("ans")
                    print(self.List)
                    self.Ans.append(self.List)
                    break


# #显示所有列
# pd.set_option('display.max_columns', None)
#
# #显示所有行
# pd.set_option('display.max_rows', None)
#
# #设置value的显示长度为100，默认为50
# pd.set_option('max_colwidth',100)
K = Classify()
Train_data = pd.read_csv('./dataset/train_class_005.csv',encoding='utf-8')
strlist = """emd_lable2
pref_city_y2_2
pref_city_y3_2
pref_city_y3_3
pref_city_y2_3
pref_city_y3_4"""
List = strlist.split("\n")

pref_air = Train_data[List]
# from sklearn.preprocessing import OrdinalEncoder
# #将文本标签转化成数字
# ordinal_encoder = OrdinalEncoder()
#
# text_encoded = ordinal_encoder.fit_transform(pref_air)
# # text_encoded2 = ordinal_encoder.fit_transform(feature2)
#
# text_encoded[:10]
# # text_encoded2[:10]
# ordinal_encoder.categories_  #查看一下映射
#
# Train_data[List] = text_encoded
# test_data[list] = text_encoded2

# Train_data.head()
List2 = []

SList = np.zeros((10,150))
SList2 = np.zeros((10,150))
for i in range(pref_air.shape[0]):
#     # x1 = pref_air.loc[i]['pax_fcny']
#     # x2 = pref_air.loc[i]['pax_tax']
#     # x3 = pref_air.loc[i]['dist_cnt_y1']
#     # x4 = pref_air.loc[i]['dist_cnt_y3']
#     # x5 = pref_air.loc[i]['dist_i_cnt_y2']
#     # x6 = pref_air.loc[i]['dist_i_cnt_y3']
#     # x7 = pref_air.loc[i]['dist_all_cnt_y2']
#     # x1 = pref_air.loc[i]['avg_dist_cnt_y1']
#     # x2 = pref_air.loc[i]['avg_dist_cnt_y3']
#     # x3 = pref_air.loc[i]['avg_dist_cnt_y2']
#     # x4 = pref_air.loc[i]['ffp_nbr']
#     # x5 = pref_air.loc[i]['tkt_3y_amt']
#     # x6 = pref_air.loc[i]['tkt_avg_interval_y2']
#     x1 = pref_air.loc[i]['tkt_avg_amt_y2']
#     x2 = pref_air.loc[i]['tkt_avg_amt_y3']
#     x3 = pref_air.loc[i]['tkt_all_amt_y3']
#     x4 = pref_air.loc[i]['tkt_i_amt_y2']
#     x5 = pref_air.loc[i]['tkt_i_amt_y3']
#     x6 = pref_air.loc[i]['pit_accu_amt_y1']
#     x7 = pref_air.loc[i]['pit_accu_air_amt']
#     x8 = pref_air.loc[i]['pit_avg_amt_y2']
#     x9 = pref_air.loc[i]['pit_avg_amt_y3']
    x1 = pref_air.loc[i]['pref_city_y2_2']
    x2 = pref_air.loc[i]['pref_city_y3_2']
    x3 = pref_air.loc[i]['pref_city_y3_3']
    x4 = pref_air.loc[i]['pref_city_y2_3']
    x5 = pref_air.loc[i]['pref_city_y3_4']

#     # z = K.country2(x,y)
#     # List2.append(z)
#     # k1 = K.Pax_fcny(x1)
#     # k2 = K.Pax_tax(x2)
#     # k3 = K.dist_cnt(x3)
#     # k4 = K.dist_cnt(x4)
#     # k5 = K.dist_i_cnt(x5)
#     # k6 = K.dist_i_cnt(x6)
#     # k7 = K.dist_all_cnt(x7)
#
#     # k1 = K.avg_dist_cnt(x1)
#     # k2 = K.avg_dist_cnt(x2)
#     # k3 = K.avg_dist_cnt(x3)
#     # k4 = K.ffp_nbr(x4)
#     # k5 = K.tkt_3y(x5)
#     # k6 = K.tkt_avg_amt(x6)
#
#     k1 = K.tkt_avg_amt2(x1)
#     k2 = K.tkt_avg_amt2(x2)
#     k3 = K.tkt_avg_amt2(x3)
#     k4 = K.tkt_avg_amt2(x4)
#     k5 = K.tkt_avg_amt2(x5)
#     k6 = K.pit_accu_amt(x6)
#     k7 = K.pit_accu_air_amt(x7)
#     k8 = K.pit_avg_amt(x8)
#     k9 = K.pit_avg_amt(x9)
#
    SList[0,x1]+=1
    SList[1, x2] += 1
    SList[2, x3] += 1
    SList[3, x4] += 1
    SList[4, x5] += 1
    # SList[5, k6] += 1
    # SList[6, k7] += 1
    # SList[7, k8] += 1
    # SList[8, k9] += 1
    # SList[6, k7] += 1
#
    if(pref_air.loc[i]['emd_lable2']>0):
        SList2[0,x1]+=1
        SList2[1, x2] += 1
        SList2[2, x3] += 1
        SList2[3, x4] += 1
        SList2[4, x5] += 1

    if(i%500 == 0):
        print(i)
for i in range(10):
    for j in range(10):
        SList[j][i] = SList2[j][i]/SList[j][i] if SList[j][i] > 0 else 0
#     # SList[1][i] = SList2[1][i]/SList[1][i] if SList[1][i] > 0 else 0
#
print(SList)
#
pref_air2 = pref_air[List]
#
for i in range(pref_air.shape[0]):
    # x1 = pref_air.loc[i]['pax_fcny']
    # x2 = pref_air.loc[i]['pax_tax']
    # x3 = pref_air.loc[i]['dist_cnt_y1']
    # x4 = pref_air.loc[i]['dist_cnt_y3']
    # x5 = pref_air.loc[i]['dist_i_cnt_y2']
    # x6 = pref_air.loc[i]['dist_i_cnt_y3']
    # x7 = pref_air.loc[i]['dist_all_cnt_y2']
    # x1 = pref_air.loc[i]['avg_dist_cnt_y1']
    # x2 = pref_air.loc[i]['avg_dist_cnt_y3']
    # x3 = pref_air.loc[i]['avg_dist_cnt_y2']
    # x4 = pref_air.loc[i]['ffp_nbr']
    # x5 = pref_air.loc[i]['tkt_3y_amt']
    # x6 = pref_air.loc[i]['tkt_avg_interval_y2']
    # x1 = pref_air.loc[i]['tkt_avg_amt_y2']
    # x2 = pref_air.loc[i]['tkt_avg_amt_y3']
    # x3 = pref_air.loc[i]['tkt_all_amt_y3']
    # x4 = pref_air.loc[i]['tkt_i_amt_y2']
    # x5 = pref_air.loc[i]['tkt_i_amt_y3']
    # x6 = pref_air.loc[i]['pit_accu_amt_y1']
    # x7 = pref_air.loc[i]['pit_accu_air_amt']
    # x8 = pref_air.loc[i]['pit_avg_amt_y2']
    # x9 = pref_air.loc[i]['pit_avg_amt_y3']
    # k1 = K.Pax_fcny(x1)
    # k2 = K.Pax_tax(x2)
    # k3 = K.dist_cnt(x3)
    # k4 = K.dist_cnt(x4)
    # k5 = K.dist_i_cnt(x5)
    # k6 = K.dist_i_cnt(x6)
    # k7 = K.dist_all_cnt(x7)
    # k1 = K.avg_dist_cnt(x1)
    # k2 = K.avg_dist_cnt(x2)
    # k3 = K.avg_dist_cnt(x3)
    # k4 = K.ffp_nbr(x4)
    # k5 = K.tkt_3y(x5)
    # k6 = K.tkt_avg_amt(x6)
    # k1 = K.tkt_avg_amt2(x1)
    # k2 = K.tkt_avg_amt2(x2)
    # k3 = K.tkt_avg_amt2(x3)
    # k4 = K.tkt_avg_amt2(x4)
    # k5 = K.tkt_avg_amt2(x5)
    # k6 = K.pit_accu_amt(x6)
    # k7 = K.pit_accu_air_amt(x7)
    # k8 = K.pit_avg_amt(x8)
    # k9 = K.pit_avg_amt(x9)
    # pref_air2.loc[i,'pax_fcny'] = SList[0][k1]
    # pref_air2.loc[i,'pax_tax'] = SList[1][k2]
    # pref_air2.loc[i,'dist_cnt_y1'] = SList[2][k3]
    # pref_air2.loc[i,'dist_cnt_y3'] = SList[3][k4]
    # pref_air2.loc[i,'dist_i_cnt_y2'] = SList[4][k5]
    # pref_air2.loc[i,'dist_i_cnt_y3'] = SList[5][k6]
    # pref_air2.loc[i,'dist_all_cnt_y2'] = SList[6][k7]
    # pref_air2.loc[i, 'avg_dist_cnt_y1'] = SList[0][k1]
    # pref_air2.loc[i, 'avg_dist_cnt_y3'] = SList[1][k2]
    # pref_air2.loc[i, 'avg_dist_cnt_y2'] = SList[2][k3]
    # pref_air2.loc[i, 'ffp_nbr'] = SList[3][k4]
    # pref_air2.loc[i, 'tkt_3y_amt'] = SList[4][k5]
    # pref_air2.loc[i, 'tkt_avg_interval_y2'] = SList[5][k6]
    # pref_air2.loc[i,'tkt_avg_amt_y2'] = SList[0][k1]
    # pref_air2.loc[i,'tkt_avg_amt_y3'] = SList[1][k2]
    # pref_air2.loc[i,'tkt_all_amt_y3'] = SList[2][k3]
    # pref_air2.loc[i,'tkt_i_amt_y2'] = SList[3][k4]
    # pref_air2.loc[i,'tkt_i_amt_y3'] = SList[4][k5]
    # pref_air2.loc[i,'pit_accu_amt_y1'] = SList[5][k6]
    # pref_air2.loc[i,'pit_accu_air_amt'] = SList[6][k7]
    # pref_air2.loc[i,'pit_avg_amt_y2'] = SList[7][k8]
    # pref_air2.loc[i,'pit_avg_amt_y3'] = SList[8][k9]
    if(i%500 == 0):
        print(i)
# List.append('seat')
# print(pref_air2)
# # Train_data['country'] = List2
# Train_data[List] = pref_air
Train_data.to_csv('./dataset/train_class_005.csv',encoding='utf-8')

