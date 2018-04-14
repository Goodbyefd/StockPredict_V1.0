#2018.3.29
#使用模型预测未来5日涨跌幅
<<<<<<< HEAD
<<<<<<< HEAD
#修改输出结果为区间估计，相应修改程序
#程序运行未现重大问题，但结果显示还是不正确，预测值总是为5%以上，检查模型训练过程序
#批量预测
=======
#预测结果严重错误
>>>>>>> parent of 831307d... 解决softmax溢出问题
=======
#预测结果严重错误
>>>>>>> parent of 831307d... 解决softmax溢出问题
import tensorflow as tf 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

#定义网络参数
inputSize = 3
hidenNodeSize = 10
<<<<<<< HEAD
<<<<<<< HEAD
outputSize = 5
batchSize = 5
learnRate = 0.001
testStart = 1002
testEnd = 1990
=======
outputSize = 1
batchSize = 5
learnRate = 0.001
testNum = 1992
>>>>>>> parent of 831307d... 解决softmax溢出问题
=======
outputSize = 1
batchSize = 5
learnRate = 0.001
testNum = 1992
>>>>>>> parent of 831307d... 解决softmax溢出问题

#导入测试数据
f = open("stockdata_v1.csv")
stockData = pd.read_csv(f)

<<<<<<< HEAD
<<<<<<< HEAD
testData = stockData.iloc[testStart:testEnd,15:18].values   #原为8：18
=======
testData = stockData.iloc[testNum,7:17].values
>>>>>>> parent of 831307d... 解决softmax溢出问题
=======
testData = stockData.iloc[testNum,7:17].values
>>>>>>> parent of 831307d... 解决softmax溢出问题

#定义网络参数
weights = {
    'in':tf.Variable(tf.random_normal([inputSize,outputSize])),
    'out':tf.Variable(tf.random_normal([hidenNodeSize,outputSize]))
}

biases = {
    'in':tf.Variable(tf.random_normal([outputSize,])),   #此定义法为行向量
    'out':tf.Variable(tf.random_normal([outputSize,]))    #此定义法为行向量
}

#定义前向过程
def bpnn_predict(train_x):
    w_in = weights['in']
    b_in = biases['in']
    w_out = weights['out']
    b_out = biases['out']
<<<<<<< HEAD
    #input_data =tf.nn.sigmoid(tf.matmul(train_x,w_in) + b_in)
    #hidenLay_out = tf.matmul(input_data,w_out) +b_out
    #nn_out = tf.nn.softmax(hidenLay_out)
    nn_out = tf.nn.softmax(tf.matmul(train_x,w_in) +b_in)
    return nn_out

def bp_predict():
    X = tf.placeholder(tf.float32,shape = [None,inputSize],name = 'inputdata')
=======
    input_data = tf.matmul(train_x1,w_in) + b_in
    #hidenLay_out = tf.nn.relu(input_data)
    nn_out = tf.matmul(input_data,w_out) +b_out
    return nn_out

def bp_predict():
    X = tf.placeholder(tf.float32,shape = [inputSize],name = 'inputdata')
    Y = tf.placeholder(tf.float32,shape = [outputSize],name = 'outputdata')
<<<<<<< HEAD
>>>>>>> parent of 831307d... 解决softmax溢出问题
=======
>>>>>>> parent of 831307d... 解决softmax溢出问题

    y_predict = bpnn_predict(X)

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        moudleFile = tf.train.latest_checkpoint('./modelsave/')
        saver.restore(sess,moudleFile)
        prob = sess.run(y_predict,feed_dict = {X:testData})
        print('未来5日涨跌幅：')
        #显示预测结果（全是4？）
        predict_y = tf.argmax(prob,1)
        print(predict_y.eval())
        """ plt.figure()
        plt.plot(list(range(len(predict_y))),predict_y,color = 'b')
        plt.show() """
bp_predict()
