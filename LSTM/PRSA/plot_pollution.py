from pandas import read_csv
from matplotlib import pyplot
# 读取数据
dataset = read_csv('pollution.csv', header=0, index_col=0)
values = dataset.values
# 选择几列进行输出
plot_column = [0,1,2,3,5,6,7]
i = 1
# 绘制每一列
pyplot.figure()
for column in plot_column:
    pyplot.subplot(len(plot_column), 1, i)
    pyplot.plot(values[:, column])
    pyplot.title(dataset.columns[column], y=0.5, loc='right')
    i += 1
pyplot.show()

