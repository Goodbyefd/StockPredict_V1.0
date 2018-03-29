#2018.3.29
#使用模型预测未来5日涨跌幅
#预测结果严重错误
import tensorflow as tf 
import numpy as np 
import pandas as pd 

#定义网络参数
inputSize = 10
hidenNodeSize = 10
outputSize = 1
batchSize = 5
learnRate = 0.001
testNum = 1992

#导入测试数据
f = open("stockdata_v1.csv")
stockData = pd.read_csv(f)

testData = stockData.iloc[testNum,7:17].values

#定义网络参数
weights = {
    'in':tf.Variable(tf.random_normal([inputSize,hidenNodeSize])),
    'out':tf.Variable(tf.random_normal([hidenNodeSize,outputSize]))
}

biases = {
    'in':tf.Variable(tf.random_normal([hidenNodeSize,])),   #此定义法为行向量
    'out':tf.Variable(tf.random_normal([outputSize,]))    #此定义法为行向量
}

#定义前向过程
def bpnn_predict(train_x):
    train_x1 = tf.reshape(train_x,shape = [1,10])           #本来只是一个行向量，重整为矩阵以得张量运算
    w_in = weights['in']
    b_in = biases['in']
    w_out = weights['out']
    b_out = biases['out']
    input_data = tf.matmul(train_x1,w_in) + b_in
    #hidenLay_out = tf.nn.relu(input_data)
    nn_out = tf.matmul(input_data,w_out) +b_out
    return nn_out

def bp_predict():
    X = tf.placeholder(tf.float32,shape = [inputSize],name = 'inputdata')
    Y = tf.placeholder(tf.float32,shape = [outputSize],name = 'outputdata')

    y_predict = bpnn_predict(X)

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        moudleFile = tf.train.latest_checkpoint('./modelsave/')
        saver.restore(sess,moudleFile)
        prob = sess.run(y_predict,feed_dict = {X:testData})
        print('未来5日涨跌幅：',prob)

bp_predict()
