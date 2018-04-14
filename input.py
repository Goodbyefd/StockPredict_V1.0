<<<<<<< HEAD
#2018.03.29  
#StockPredict V1.1.1
<<<<<<< HEAD
<<<<<<< HEAD
#原直接使用softmax对神经网络输出进行处理，会导致log(predict_y)溢出为non
#将loss改为tf.nn.softmax_cross_entropy_with_logits后程序运行正常
#但收敛进9即无法再下降，预测值总是为5%以上
#添加中间层relu激活函数，相应梯度下降是否要修改？
#收敛得很不正确，中间阵使用使用relu显示部分已达e+8
=======
=======
>>>>>>> parent of 831307d... 解决softmax溢出问题
#输出结果使用区间方式展示，激活函数使用softmax，
#修改训练数据标准化方式，仅对feature进行标准化，label不再进行标准化
#修改loss定义以及迭代化器为递度下降
#程序运行无报错，但loss持续显示non,无法收敛
#查看一下是否为softmax使用方式错误所致
<<<<<<< HEAD
>>>>>>> parent of 831307d... 解决softmax溢出问题
=======
>>>>>>> parent of 831307d... 解决softmax溢出问题
=======
#2018.03.28  
#StockPredict V1.0.1
##已使用批次处理问题
#并保存每次模型参数
#但loss总是在0.21左右无法下降

>>>>>>> parent of 20e15aa... StockPredict V1.1.1

import tensorflow as tf 
import numpy as np 
import pandas as pd 

#定义网络参数
inputSize = 3
hidenNodeSize = 10
outputSize = 1
batchSize = 5
learnRate = 0.001

#导入测试数据
f = open("stockdata_v1.csv")
stockData = pd.read_csv(f)
#取第2-9列数据
<<<<<<< HEAD
data = stockData.iloc[:,15:23].values   #原为8：23
=======
data = stockData.iloc[:,7:18].values
>>>>>>> parent of 20e15aa... StockPredict V1.1.1

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
    'in':tf.Variable(tf.random_normal([inputSize,outputSize],seed = 1,stddev = 2)),
    'out':tf.Variable(tf.random_normal([hidenNodeSize,outputSize],seed = 1,stddev = 2))
}

biases = {
    'in':tf.Variable(tf.zeros([outputSize,])),   #此定义法为行向量
    'out':tf.Variable(tf.zeros([outputSize,]))    
}


#定义前向过程
def bpnn(train_x):
    #train_x1 = tf.reshape(train_x,shape = [None,10])           #本来只是一个行向量，重整为矩阵以得张量运算
    w_in = weights['in']
    b_in = biases['in']
    w_out = weights['out']
    b_out = biases['out']
<<<<<<< HEAD
    #input_data =tf.nn.sigmoid(tf.matmul(train_x,w_in) + b_in)
    nn_out = tf.nn.softmax(tf.matmul(train_x,w_in) +b_in)   
    return nn_out      #这个nn_out很可能很大？
=======
    input_data = tf.matmul(train_x,w_in) + b_in
    #hidenLay_out = tf.nn.relu(input_data)
    nn_out = tf.matmul(input_data,w_out) +b_out
    return nn_out
>>>>>>> parent of 831307d... 解决softmax溢出问题

#定义训练过程
def nn_train(train_x,train_y):
    #定义数据承载空间
    X = tf.placeholder(tf.float32,shape = [None,inputSize],name = 'inputdata')
    Y = tf.placeholder(tf.float32,shape = [None,outputSize],name = 'outputdata')

    predict_y = bpnn(X)
    #定义损失函数和优化器
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    #loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = Y,logits = predict_y))
    loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(predict_y + 1e-10), reduction_indices=1))
=======
    #loss = tf.reduce_mean(tf.square(tf.reshape(predict_y,[-1]) - tf.reshape(Y,[-1])))
    loss = tf.reduce_mean(-tf.reduce_sum(predict_y * tf.log(Y),axis = 1))
>>>>>>> parent of 831307d... 解决softmax溢出问题
=======
    #loss = tf.reduce_mean(tf.square(tf.reshape(predict_y,[-1]) - tf.reshape(Y,[-1])))
    loss = tf.reduce_mean(-tf.reduce_sum(predict_y * tf.log(Y),axis = 1))
>>>>>>> parent of 831307d... 解决softmax溢出问题
    train_op = tf.train.GradientDescentOptimizer(learnRate).minimize(loss)
=======
    loss = tf.reduce_mean(tf.square(tf.reshape(predict_y,[-1]) - tf.reshape(Y,[-1])))
    train_op = tf.train.AdamOptimizer(learnRate).minimize(loss)
>>>>>>> parent of 20e15aa... StockPredict V1.1.1
    #定义模型保存器
    modelSaver = tf.train.Saver(tf.global_variables(),max_to_keep=15)
    #定义会话过程
    with tf.Session() as sess: 
        #先初始化所有变量
        sess.run(tf.global_variables_initializer())

        for i in range(2000):  #训练次数
            for step in range(0,len(train_x) - batchSize,batchSize): #每次训练进度
<<<<<<< HEAD
<<<<<<< HEAD
                _,loss_ = sess.run([train_op,loss],feed_dict = {X:train_x[step:step + batchSize,:],Y:train_y[step:step + batchSize,:]})
=======
                loss_ = sess.run([train_op,loss],feed_dict = {X:train_x[step:step + batchSize,:],Y:train_y[step:step + batchSize,:]})
>>>>>>> parent of 831307d... 解决softmax溢出问题
=======
                loss_ = sess.run([train_op,loss],feed_dict = {X:train_x[step:step + batchSize,:],Y:train_y[step:step + batchSize,:]})
>>>>>>> parent of 831307d... 解决softmax溢出问题
            print(i,loss_)
            #保存模型变量参数
            if(i%200 == 0):
                print('保存模型：',modelSaver.save(sess,'./modelsave/StockPredict.model',global_step=i))

train_x,train_y = GetTrainData(batchSize,17,1989)
nn_train(train_x,train_y)

