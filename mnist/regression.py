# -*- coding: utf-8 -*-
# @Time : 2020/3/25 22:34
# @Author : fc
# @File : regression.py
import os
import tensorflow as tf
import input_data
import model
# data=input_data.read_data_sets('MNIST_data',one_hot=True)
MNIST_data_folder="E:\project\python_workspace\mnist_testdemo\mnist\MNIST_data"
data=input_data.read_data_sets(MNIST_data_folder,one_hot=True)
# 线性



# 定义完模型之后引入模型和训练：

# create model

with tf.variable_scope('regression'):  # 命名

    x = tf.placeholder(tf.float32, [None, 784])  # x:待用户输入，用一个占位符，placeholder的第一个参数是类型，第二个是张量，其中的784是和model中对应

    y, variables = model.regression(x)

# train

y_ = tf.placeholder('float', [None, 10])

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))  # 训练的交叉熵

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(
    cross_entropy)  # 训练步骤，GradientDescentOptimizer(0.01)：一个优化器，设置步长为0.01

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # 预测

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 准确率，tf.cast(correct_prediction, tf.float32)：转换格式

# 参数进行保存

saver = tf.train.Saver(variables)

# 开始训练

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 把全部的参数放进来，进行全局初始化

    for _ in range(1000):  # 训练1000次

        batch_xs, batch_ys = data.train.next_batch(100)

        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})  # feed_dict：喂参数，x放的batch_xs,y_放的batch_ys

    print((sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels})))  # 打印测试集的图像和标签

    path = saver.save(

        # 把数据存进去，把这个模型的名字存成regression.ckpt，在这里注意data文件夹的创建

        sess, os.path.join(os.path.dirname(__file__), 'data', 'regression.ckpt'),

        write_meta_graph=False, write_state=False  # 写到图中

    )  # 把数据或者说是参数或者说是模型存起来

    print('Saved:', path)  # 把保存模型的路径打印出来