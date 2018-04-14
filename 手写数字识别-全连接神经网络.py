# -*- coding: utf-8 -*-  
""" 
@author: tz_zs(5_2) 
 
MNIST手写体数字识别 
"""  
  
import tensorflow as tf  
from tensorflow.examples.tutorials.mnist import input_data  
  
# MNIST 数据集相关的常数  
INPUT_NODE = 784  # 输入层为28*28的像素  
OUTPUT_NODE = 10  # 输出层0~9有10类  
  
# 配置神经网络的参数  
LAYER1_NODE = 500  # 隐藏层节点数  
  
BATCH_SIZE = 100  # batch的大小  
  
LEARNING_RATE_BASE = 0.8  # 基础的学习率  
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率  
REGULARIZATION_RATE = 0.0001  # 正则化项的系数  
TRAINING_STEPS = 30000  # 训练轮数  
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均的衰减率  
  
  
# 定义神经网络的结构  
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):  
    # 当没有提供滑动平均类时，直接使用参数当前的值  
    if avg_class == None:  
        # 计算隐藏层的前向传播结果，ReLU激活函数  
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)  
        # 计算输出层的前向传播结果（计算损失函数时，会一并进行softmax运输，在这里不进行softmax回归）  
        return tf.matmul(layer1, weights2) + biases2  
    else:  
        # 需要先使用滑动平均值计算出参数  
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))  
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)  
  
  
# 定义训练模型的操作  
def train(mnist):  
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')  
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')  
  
    # 生成隐藏层的参数  
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))  
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))  
  
    # 生成输出层的参数  
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))  
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))  
  
    # 定义计算当前参数下，神经网络前向传播的结果。  
    y = inference(x, None, weights1, biases1, weights2, biases2)  
  
    # 定义存储训练轮数的变量  
    global_step = tf.Variable(0, trainable=False)  
  
    # 初始化滑动平均对象  
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)  
  
    # 定义滑动平均的操作  
    variable_averages_op = variable_averages.apply(tf.trainable_variables())  
  
    # 定义计算使用了滑动平均之后的前向传播结果  
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)  
  
    # 损失函数  
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)  
    cross_entropy_mean = tf.reduce_mean(cross_entropy)  
    # L2  
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)  
    regularization = regularizer(weights1) + regularizer(weights2)  
    loss = cross_entropy_mean + regularization  
  
    # 学习率  
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE,  
                                               LEARNING_RATE_DECAY)  
  
    # 优化算法  
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)  
  
    # 训练的操作  
    with tf.control_dependencies([train_step, variable_averages_op]):  
        train_op = tf.no_op(name='train')  
  
    # 检验 准确度  
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))  
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
  
    with tf.Session() as sess:  
        tf.global_variables_initializer().run()  
  
        # 准备数据  
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}  
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}  
  
        # 迭代的训练神经网络  
        for i in range(TRAINING_STEPS):  
            # 每1000轮，使用验证数据测试一次  
            if i % 1000 == 0:  
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)  
                print("After %d training step(s), validation accuracy "  
                      "using average model is %g " % (i, validate_acc))  
  
            # 训练的数据  
            xs, ys = mnist.train.next_batch(BATCH_SIZE)  
            sess.run(train_op, feed_dict={x: xs, y_: ys})  
  
        # 测试最终的准确率  
        test_acc = sess.run(accuracy, feed_dict=test_feed)  
        print("After %d training step(s), test accuracy using average model is %g " % (TRAINING_STEPS, test_acc))  
  
  
# 主程序入口  
def main(argv=None):  
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)  
    train(mnist)  
  
  
if __name__ == '__main__':  
    tf.app.run()  