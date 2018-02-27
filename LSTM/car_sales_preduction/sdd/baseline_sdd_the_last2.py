import pandas as pd
import matplotlib.pyplot as plt

train=pd.read_csv("yancheng_train_20171226.csv")
test=pd.read_csv("yancheng_testA_20171225.csv")
test_class_id=test['class_id'].values   # id
sale_train_data=[[] for i in range(len(test_class_id))] # 数据从0到总数的编号
sale_train_data_x_time=[]
count=0

all_time_axis=[201201+i for i in range(12)] # 月份编号
for i in range(12):
   all_time_axis.append(201301+i)
for i in range(12):
   all_time_axis.append(201401+i)
for i in range(12):
   all_time_axis.append(201501+i)
for i in range(12):
   all_time_axis.append(201601+i)
for i in range(10):
   all_time_axis.append(201701+i)

for _class_id in test_class_id:
   temp_train=train[train['class_id']==_class_id]
   print(temp_train)
  # temp_train=temp_train.sort_values(by='sale_date')
   time_list=set(list(temp_train['sale_date'].values))
   print(time_list)
   time_list=list(time_list)
   print(time_list)
   time_list=sorted(time_list)
   print(time_list)
   #print(time_list)
   print("shape,count,list",temp_train.shape,count,len(time_list))   
   sale_train_data_x_time.append(time_list)
   for _time in all_time_axis:
     if(_time in time_list):
       temp_train_bytime=temp_train[temp_train['sale_date']==_time]
       temp_sale_num=temp_train_bytime['sale_quantity'].values
       sale_sum=temp_sale_num.sum()
       sale_train_data[count].append(sale_sum)
     else:
       sale_train_data[count].append(0) #fill nan   
   count=count+1

   exit()


test_predict_value=[]
for i in range(len(sale_train_data)):
     predict_values=1.8*sale_train_data[i][-1]
     test_predict_value.append(int(predict_values))

import numpy as np
test_predict_value=np.array(test_predict_value)
test['predict_quantity']=test_predict_value
test.to_csv('./Result/sdd_K=1.8_the_last_1.csv',index=False,encoding='utf-8')




