from pandas import read_csv
from datetime import datetime

def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')

dataset = read_csv('PRSA_data_2010.1.1-2014.12.31.csv', parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
dataset.drop('No', axis=1, inplace=True)
# 确定列名
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'data'
# 将缺少的数据（Na）用0替换
dataset['pollution'].fillna(0, inplace=True)
# 去掉前24列无用的数据
dataset = dataset[24:]
# 显示前五个，来表示处理后的效果
print(dataset.head(5))
# 存储CSV
dataset.to_csv('pollution.csv')



