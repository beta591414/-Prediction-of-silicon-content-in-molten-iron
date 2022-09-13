import pandas as pd
import numpy as np
from matplotlib import pyplot
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import concatenate
from collections import Counter

trainlist = pd.read_csv('traindata.csv', index_col='time', encoding="gbk")
testlist1 = pd.read_csv('test7.csv', index_col='time', encoding="gbk")
testlist2 = pd.read_csv('test12.csv', index_col='time', encoding="gbk")
test_y1 = pd.read_csv('test7_y.csv', index_col='time', encoding="gbk")
test_y2 = pd.read_csv('test12_y.csv', index_col='time', encoding="gbk")
groups = list(range(0,37))
#print(trainlist.shape[0])
#print(trainlist.iloc[:, [36]]) #选取dataframe的第37列
"""
pyplot.rcParams['font.family']=['SimHei']
pyplot.title('变量趋势变化')
pyplot.subplot(2,1,1)
pyplot.plot(trainlist['冷风压力'], label = '冷风压力', color='g')
pyplot.legend()
pyplot.subplot(2,1,2)
pyplot.plot(trainlist['冷风压力2'], label = '冷风压力2', color='b')
pyplot.legend()
pyplot.show()
"""

#特征工程，构造变量
trainlist['顶压mean'] = (trainlist['顶压'] + trainlist['顶压2'] + trainlist['顶压3'] + trainlist['顶压4'] ) / 4
trainlist['冷风压力mean'] = (trainlist['冷风压力'] + trainlist['冷风压力2'] ) / 2
trainlist['热风压力mean'] = (trainlist['热风压力'] + trainlist['热风压力2'] ) / 2
trainlist['顶温mean'] = (trainlist['顶温东北'] + trainlist['顶温西南'] + trainlist['顶温西北'] + trainlist['顶温东南']) / 4
trainlist.insert(40, 'last', trainlist.pop('last'))
trainlist.insert(40, 'label', trainlist.pop('label'))
trainlist.drop(columns= ['顶压', '顶压2', '顶压3', '顶压4', '冷风压力', '冷风压力2', '热风压力', '热风压力2', '顶温东北', '顶温西南', '顶温西北', '顶温东南', '设定喷煤量', '本小时实际喷煤量', '上小时实际喷煤量'], inplace=True)

testlist1['顶压mean'] = (testlist1['顶压'] + testlist1['顶压2'] + testlist1['顶压3'] + testlist1['顶压4'] ) / 4
testlist1['冷风压力mean'] = (testlist1['冷风压力'] + testlist1['冷风压力2'] ) / 2
testlist1['热风压力mean'] = (testlist1['热风压力'] + testlist1['热风压力2'] ) / 2
testlist1['顶温mean'] = (testlist1['顶温东北'] + testlist1['顶温西南'] + testlist1['顶温西北'] + testlist1['顶温东南']) / 4
testlist1.drop(columns= ['顶压', '顶压2', '顶压3', '顶压4', '冷风压力', '冷风压力2', '热风压力', '热风压力2', '顶温东北', '顶温西南', '顶温西北', '顶温东南', '设定喷煤量', '本小时实际喷煤量', '上小时实际喷煤量'], inplace=True)
testlist1.insert(24, 'last', testlist1.pop('last'))
testlist1['label'] = test_y1['label']

testlist2['顶压mean'] = (testlist2['顶压'] + testlist2['顶压2'] + testlist2['顶压3'] + testlist2['顶压4'] ) / 4
testlist2['冷风压力mean'] = (testlist2['冷风压力'] + testlist2['冷风压力2'] ) / 2
testlist2['热风压力mean'] = (testlist2['热风压力'] + testlist2['热风压力2'] ) / 2
testlist2['顶温mean'] = (testlist2['顶温东北'] + testlist2['顶温西南'] + testlist2['顶温西北'] + testlist2['顶温东南']) / 4
testlist2.drop(columns= ['顶压', '顶压2', '顶压3', '顶压4', '冷风压力', '冷风压力2', '热风压力', '热风压力2', '顶温东北', '顶温西南', '顶温西北', '顶温东南', '设定喷煤量', '本小时实际喷煤量', '上小时实际喷煤量'], inplace=True)
testlist2.insert(24, 'last', testlist2.pop('last'))
testlist2['label'] = test_y2['label']

#异常值处理，箱型图分析法
def outliners(data, col):
    def box_plot_outliners(data_ser):
        IQR = data_ser.quantile(0.75)-data_ser.quantile(0.25)
        val_low = data_ser.quantile(0.25) - IQR * 1.5
        val_up = data_ser.quantile(0.75) + IQR * 1.5
        rule_low = (data_ser < val_low)
        rule_up = (data_ser > val_up)
        print(rule_low)
        return rule_low, rule_up, val_low, val_up
    data_n = data.copy()
    data_series = data_n[col]
    rule_low, rule_up, val_low, val_up = box_plot_outliners(data_series)
    data_n[col].loc[rule_up] = val_up
    data_n[col].loc[rule_low] = val_low
    return data_n

col_names = list(trainlist)
del col_names[24:26]
trainlist.iloc[:, 2:5].boxplot()
pyplot.show()  
for col_name in col_names:
    trainlist = outliners(trainlist, col_name).copy()
    testlist1 = outliners(testlist1, col_name).copy()
    testlist2 = outliners(testlist2, col_name).copy()
trainlist.iloc[:, 2:5].boxplot()
pyplot.show()  


#将训练集划分为两份，分别对应着两份测试集（7月与12月）
trainlist1 = trainlist.iloc[:3237,:]
trainlist2 = trainlist.iloc[3237:,:]
#print(testlist1)
#print(trainlist2)

#归一化（除了label值保持不变），并从dataframe格式变成数组格式
scaler = MinMaxScaler(feature_range=(0,1))
trainlist1.iloc[:, :-2] = scaler.fit_transform(trainlist1.iloc[:, :-2]).copy()
trainlist2.iloc[:, :-2] = scaler.fit_transform(trainlist2.iloc[:, :-2]).copy()
testlist1.iloc[:, :-2] = scaler.fit_transform(testlist1.iloc[:, :-2]).copy()
testlist2.iloc[:, :-2] = scaler.fit_transform(testlist2.iloc[:, :-2]).copy()


trainlist1.to_csv('train1.csv')
trainlist2.to_csv('train2.csv')
testlist1.to_csv('test1.csv')
testlist2.to_csv('test2.csv')

trainlist1 = np.array(trainlist1)
trainlist2 = np.array(trainlist2)
testlist1 = np.array(testlist1)
testlist2 = np.array(testlist2)
"""
#构建模型,先构建第一个
train_X1, train_y1= trainlist1[:, :-1], trainlist1[:, -1]
test_X1, test_y1 = testlist1[:, :-1], testlist1[:, -1]

train_X1 = train_X1.reshape((train_X1.shape[0], 1, train_X1.shape[1]))
test_X1 = test_X1.reshape((test_X1.shape[0], 1, test_X1.shape[1]))

model = Sequential()
model.add(LSTM(32, input_shape=(train_X1.shape[1], train_X1.shape[2]), return_sequences=True))
model.add(LSTM(32))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='Adam')
# fit network
history = model.fit(train_X1, train_y1, epochs=10, batch_size=1, validation_data=(test_X1, test_y1), verbose=2, shuffle=False)
# plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()

yhat1 = model.predict(test_X1)

test_X1 = test_X1.reshape((test_X1.shape[0], test_X1.shape[2]))
test_y1 = test_y1.reshape((len(test_y1), 1))

mse = mean_squared_error(test_y1, yhat1)
print('Test1 MSE: %.3f' % mse)

pyplot.rcParams['font.family']=['SimHei']
pyplot.title('test1')
pyplot.plot(test_y1,label = '实际值')
pyplot.plot(yhat1, label = '预测值')
pyplot.legend()
pyplot.show()
"""
#同理构建第二个
train_X2, train_y2= trainlist2[:, :-1], trainlist2[:, -1]
test_X2, test_y2 = testlist2[:, :-1], testlist2[:, -1]

train_X2 = train_X2.reshape((train_X2.shape[0], 1, train_X2.shape[1]))
test_X2 = test_X2.reshape((test_X2.shape[0], 1, test_X2.shape[1]))

model = Sequential()
model.add(LSTM(32, input_shape=(train_X2.shape[1], train_X2.shape[2]), return_sequences=True))
model.add(LSTM(32))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='Adam')
# fit network
history = model.fit(train_X2, train_y2, epochs=10, batch_size=1, validation_data=(test_X2, test_y2), verbose=2, shuffle=False)
# plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()

yhat2 = model.predict(test_X2)

test_X2 = test_X2.reshape((test_X2.shape[0], test_X2.shape[2]))
test_y2 = test_y2.reshape((len(test_y2), 1))

mse = mean_squared_error(test_y2, yhat2)
print('Test2 MSE: %.3f' % mse)

pyplot.rcParams['font.family']=['SimHei']
pyplot.title('test2')
pyplot.plot(test_y2, label = '实际值')
pyplot.plot(yhat2, label = '预测值')
pyplot.legend()
pyplot.show()