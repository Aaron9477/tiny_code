import os
import gc
import time
import pickle
import xgboost as xgb
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime
import operator
import gc
from sklearn.cross_validation import KFold
from sklearn import cross_validation
import lightgbm as lgb

traindata = pd.read_csv('traindata.csv')
testdata = pd.read_csv('testdata.csv')

data = traindata.groupby(['sale_date','class_id'])['sale_quantity'].sum().reset_index() # 按照时间和品牌分组，并对销量求和
data.columns=['sale_date','class_id','sum_quan']    # 列重命名？？？

# 取某个月的所有牌子的总销量数据
def get_traindata(month):
    data = traindata.groupby(['sale_date', 'class_id'])['sale_quantity'].sum().reset_index()
    data.columns = ['sale_date', 'class_id', 'sum_sale']
    data = data[(data.sale_date==2017*100+month)]   # 设定特定的时间
    return data

def cal_day(month,k):   # 返回month前k个月的月份数
    if(month-k>=1):
        return 2017*100+(month-k)
    else:
        return 2016*100+(12+(month-k-1))
def get_feature1(month,k):
    all = data[['class_id']].drop_duplicates()  # 去除没数据的项？？？？去除复制品？？？

    for i in range(1,k+1):
        data_t = data[data.sale_date==cal_day(month,i)] # 取该月的所有信息
        data_t.columns=['sale_date','class_id','pre_'+str(i)]
        data_t = data_t[['class_id','pre_'+str(i)]] # 只取品牌和对应销量 对应销量的时间为pre_i个月 即i月
        all = pd.merge(all,data_t,how='left',on=['class_id'])   # 根据销量进行表的合并

    return all

# 交叉验证 取后五个月分别作为测试集 只包含一个月的结果
traindata_month_7 = get_traindata(7)
traindata_month_8 = get_traindata(8)
traindata_month_9 = get_traindata(9)
traindata_month_10 = get_traindata(10)
traindata_month_11 = testdata
# 得到这五个月的对应训练集  包含十二个月的结果
traindata_month_feature_7 = pd.DataFrame(get_feature1(7,12))
traindata_month_feature_8 = pd.DataFrame(get_feature1(8,12))
traindata_month_feature_9 = pd.DataFrame(get_feature1(9,12))
traindata_month_feature_10 = pd.DataFrame(get_feature1(10,12))
traindata_month_feature_11 = pd.DataFrame(get_feature1(11,12))


# 将训练集和测试集合并 交叉验证
traindata_month_7 = pd.merge(traindata_month_7,traindata_month_feature_7,how='left',on=['class_id'])
traindata_month_8 = pd.merge(traindata_month_8,traindata_month_feature_8,how='left',on=['class_id'])
traindata_month_9 = pd.merge(traindata_month_9,traindata_month_feature_9,how='left',on=['class_id'])
traindata_month_10 = pd.merge(traindata_month_10,traindata_month_feature_10,how='left',on=['class_id'])
traindata_month_11 = pd.merge(traindata_month_11,traindata_month_feature_11,how='left',on=['class_id'])

print(traindata_month_11)
exit()


traindata_month = pd.concat([traindata_month_9])    # 只训练第九个月的？？？？？？？？？？？？？？？
feature=[]
for i in range(1,13):
    feature.append('pre_'+str(i))

# 第十个月测试集 九月及其之前的训练集？？？？
train1 = xgb.DMatrix(traindata_month[feature], label=traindata_month['sum_sale'])
test1 = xgb.DMatrix(traindata_month_10[feature], label=traindata_month_10['sum_sale'])
param = {
    'objective': 'reg:linear',
    'eta': 0.05,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'silent': 1,
    'verbose_eval': True,
    'eval_metric': 'rmse',
    'seed': 666666,
    'max_depth': 3,

}


evallist = [(train1, 'train'), (test1, 'eval')]

bst = xgb.train(param, train1, 1000, evallist,verbose_eval=30,early_stopping_rounds=30)
importance = bst.get_fscore()   #？？？？？？？？？？？
importance = sorted(importance.items(), key=operator.itemgetter(1)) # 排序？？？？？
print(importance)


traindata_month_10['pre'] = bst.predict(test1)
traindata_month_10.to_csv('watch.csv',index = False)

train1 = xgb.DMatrix(traindata_month_10[feature], label=traindata_month_10['sum_sale'])
test1 = xgb.DMatrix(traindata_month_11[feature])
evallist = [(train1, 'train')]  # 只训练得到预测数据，没有进行评测
bst = xgb.train(param, train1, 55, evallist,verbose_eval=10,early_stopping_rounds=30)
traindata_month_11['predict_quantity'] = bst.predict(test1)
traindata_month_11[['predict_date','class_id','predict_quantity']].to_csv('result.csv',index = False)   # predict_date是测试集自带的

