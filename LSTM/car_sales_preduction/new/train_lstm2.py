import numpy as np
import pandas as pd
import math
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


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

# X is the number of passengers at a given time (t) and Y is the number of passengers at the next time (t + 1).
# convert an array of values into a dataset matrix
# look_back是下一数据和之前多少个数据有联系
def create_dataset(dataset, look_back=3):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):   # 此处没有使用最后三个预测未来的量 因为没有对应的Y
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print (dataX)
    print (dataY)
    return np.array(dataX), np.array(dataY)

traindata_month_feature_11 = pd.DataFrame(get_feature1(11,18))  # 11月之前18个月的
dataset_origin = traindata_month_feature_11.iloc[:,::-1].fillna(0) # 将Nan变为0，否则不能归一化
# print(dataset)
# print(dataset.values)

# 取第一行作为例子，此处需要循环
dataset = dataset_origin.values[0,:-1] # 反向，因为LSTM是按时序进行的
print(dataset)
# tmp2 = []
# for i in tmp:
#     tmp2.append([i,])
dataset = np.reshape(dataset,(len(dataset),1)) # 去掉品牌项的，并转为类似于[[1],[2],[3]]二维形式
print(dataset)
# print(dataset.iloc[120,:].values)
# 归一化
scaler = MinMaxScaler(feature_range=(0, 1)) # 归一化[0,1]之间的方法定义
dataset_nor = scaler.fit_transform(dataset)    # 归一化之后的单一月份
print(dataset_nor)

look_back = 3   # LSTM和之前多少个时序相关
trainX, trainY = create_dataset(dataset_nor, look_back)

# 投入到 LSTM 的 X 需要有这样的结构： [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

model = Sequential()
model.add(LSTM(50, input_shape=(1, look_back))) # 隐藏层50个神经元 输入是1Xlook_back
model.add(Dense(1)) # 输出
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# 预测
trainPredict = model.predict(trainX)


# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)   # 反转换
trainY = scaler.inverse_transform([trainY]) # 反转换

# print(trainPredict)
# print(trainY)
# exit()

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))    # 误差定义为差方和
print('Train Score: %.2f RMSE' % (trainScore))
print(trainPredict) # 输出预测值
print(dataset_origin.values[0,:])  # 输出实际值

# 预测11月的结果
predict_set = np.array([dataset_nor[-3:,0]])    # 输入，注意定义方式是[[1,2,3]]

print(predict_set)
predict = model.predict(np.reshape(predict_set, (predict_set.shape[0], 1, predict_set.shape[1])))
result = scaler.inverse_transform(predict)
print(result)

exit()







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


# class_id和十二个月去Nan的值
traindata_month_11_fillna = traindata_month_11.iloc[:,[1,3,4,5,6,7,8,9,10,11,12,13,14]].fillna(0)
traindata_month_11 = pd.merge(traindata_month_11.iloc[:,:2], traindata_month_11_fillna, how='left', on=['class_id'])
dataset = traindata_month_11.iloc[:,2:]
print(dataset)
# print(traindata_month_11.values)
exit()

# dataset = traindata_month_11.iloc[0,:]
# print(dataset)
# exit()
# normalize the dataset 归一化
dataset = traindata_month_11
scaler = MinMaxScaler(feature_range=(0, 1)) # 到[0,1]之间
dataset = scaler.fit_transform(dataset)
print(dataset)