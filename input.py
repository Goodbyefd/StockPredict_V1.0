import tensorflow as tf 
import numpy as np 
import pandas as pd 

#定义网络参数
inputSize = 10
hidenNodeSize = 15
outputSize = 1
batchSize = 1
learnRate = 0.03

#导入测试数据
f = open("stockdata_v1.csv")
stockData = pd.read_csv(f)
#取第2-9列数据
data = stockData.iloc[:,7:18].values

#构造训练数据
def GetTrainData(batchSize,train_begin,train_end):
    datatrain = data[train_begin:train_end,:]
    #标准化输入输出数据，按行平均及求方差
    normalized_traindata = (datatrain - np.mean(datatrain,axis = 0))/np.std(datatrain,axis = 0) 
    train_x = normalized_traindata[:,:inputSize]               
    train_y = normalized_traindata[:,inputSize,np.newaxis]     #需要增加维度，原本只是一个列向量，转换为一个一列矩阵，后每行作为一个行向量送入预定空间
    return train_x,train_y

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
def bpnn(train_x):
    train_x1 = tf.reshape(train_x,shape = [1,10])           #本来只是一个行向量，重整为矩阵以得张量运算
    w_in = weights['in']
    b_in = biases['in']
    w_out = weights['out']
    b_out = biases['out']
    input_data = tf.matmul(train_x1,w_in) + b_in
    #hidenLay_out = tf.nn.relu(input_data)
    nn_out = tf.matmul(input_data,w_out) +b_out
    return nn_out

#定义训练过程
def nn_train(train_x,train_y):
    #定义数据承载空间
    X = tf.placeholder(tf.float32,shape = [inputSize],name = 'inputdata')
    Y = tf.placeholder(tf.float32,shape = [outputSize],name = 'outputdata')

    predict_y = bpnn(X)
    #定义损失函数和优化器
    loss = tf.reduce_mean(tf.square(predict_y - Y))
    train_op = tf.train.AdamOptimizer(learnRate).minimize(loss)
    #定义会话过程
    with tf.Session() as sess: 
        #先初始化所有变量
        sess.run(tf.global_variables_initializer())

        for i in range(2000):  #训练次数
            for step in range(len(train_x)): #每次训练进度
                loss_ = sess.run([train_op,loss],feed_dict = {X:train_x[step],Y:train_y[step]})
                print(i,loss_)

train_x,train_y = GetTrainData(batchSize,16,1989)
nn_train(train_x,train_y)
#待解决批次处理问题
#每次训练完一次后，重新开始就不收敛了？还是要做批训练