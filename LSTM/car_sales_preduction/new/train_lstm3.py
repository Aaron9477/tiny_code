import numpy as np
import pandas as pd
import math
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
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
dataset_all = traindata_month_feature_11.iloc[:, ::-1].fillna(0) # 将Nan变为0，否则不能归一化

output = pd.DataFrame(columns=["predict_date","class_id","predict_quantity"])  # 定义新表，存储结果用于输出

# 取第一行作为例子，此处需要循环
for i in range(dataset_all.shape[0]):
    dataset_origin = dataset_all.values[i, :-1] # 反向，因为LSTM是按时序进行的，并去掉品牌项
    dataset = np.reshape(dataset_origin,(len(dataset_origin),1)) # 转为类似于[[1],[2],[3]]二维形式
    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1)) # 归一化[0,1]之间的方法定义
    dataset_nor = scaler.fit_transform(dataset)    # 归一化之后的单一月份

    look_back = 3   # LSTM和之前多少个时序相关
    trainX, trainY = create_dataset(dataset_nor, look_back)

    # 投入到 LSTM 的 X 需要有这样的结构： [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

    # 网络定义
    model = Sequential()
    model.add(LSTM(50, input_shape=(1, look_back))) # 隐藏层50个神经元 输入是1Xlook_back
    model.add(Dense(1)) # 输出
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    # 预测
    trainPredict = model.predict(trainX)

    # 转换为真实值
    trainPredict = scaler.inverse_transform(trainPredict)   # 反转换
    trainY = scaler.inverse_transform([trainY]) # 反转换

    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))    # 误差定义为差方和
    print('Train Score: %.2f RMSE' % (trainScore))
    # print(trainPredict) # 输出训练集预测值
    # print(dataset_all.values[0, :])  # 输出训练集实际值

    # 预测11月的结果 取最后look_back个元素
    predict_set = np.array([dataset_nor[-look_back:,0]])    # 输入，注意定义格式是[[1,2,3]]

    predict = model.predict(np.reshape(predict_set, (predict_set.shape[0], 1, predict_set.shape[1])))
    result = scaler.inverse_transform(predict)
    print(result)

    new = pd.DataFrame({"predict_date":['201711'], "class_id":[str(int(dataset_all.values[i, -1]))], "predict_quantity":[str(result[0][0])]}, index=[str(i)])
    # new = pd.DataFrame([['201712', str(dataset_all.values[i, -1]), str(result[0][0])]], columns=["predict_date", "class_id", "predict_quantity"])
    output = output.append(new, ignore_index=True)
    print(output)

output[['predict_date','class_id','predict_quantity']].to_csv('result.csv',index = False)   # predict_date是测试集自带的
