# -*- coding: utf-8 -*-
# @Time : 2020/3/29 15:42
# @Author : fc
# @File : model.py
# 定义完模型之后引入模型和训练：

# create model
import tensorflow as tf
# Y = W * x + b
def regression(x):
    W = tf.Variable(tf.zeros([784, 10], name='W'))  # 784 * 10的二维数组

    b = tf.Variable(tf.zeros([10]), name='b')  # 一维数组里面放10个值

    y = tf.nn.softmax(tf.matmul(x, W) + b)  # softmax做简单的线性运算
    return y, [W, b]

# 卷积模型:多层卷积

def convolutional(x, keep_prob):
    def conv2d(x, W):   # 定义一个2*2的卷积
     # return tf.nn.conv2d([1, 1, 1, 1], padding='SAME')
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    def max_pool_2x2(x):  # 定义一个2*2的池化层
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    def weight_variable(shape):  # 定义一个权重变量

        initial = tf.truncated_normal(shape, stddev=0.1)

        return tf.Variable(initial)
    def bias_variable(shape):   # 定义一个偏置项变量

        initial = tf.constant(0.1, shape=shape)

        return tf.Variable(initial)
    # 定义卷积层，第一层：

    x_image = tf.reshape(x, [-1, 28, 28, 1])  # 定义图像

    W_conv1 = weight_variable([5, 5, 1, 32])  # 权重

    b_conv1 = bias_variable([32])  # 偏置项

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 定义卷积

    h_pool1 = max_pool_2x2(h_conv1)  # 定义池化

    # 第二层实际上和第一层是一样的
    W_conv2 = weight_variable([5, 5, 32, 64])  # 权重

    b_conv2 = bias_variable([64])  # 偏置项

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 定义卷积

    h_pool2 = max_pool_2x2(h_conv2)  # 定义池化
    # full connection

    W_fc1 = weight_variable([7 * 7 * 64, 1024])

    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])      # 全连接层的池化

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # dropout 可以扔掉一些值，防止过拟合

    W_fc2 = weight_variable([1024, 10])

    b_fc2 = bias_variable([10])

    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return y, [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]