# -*- coding: utf-8 -*-
# @Time : 2020/3/29 15:55
# @Author : fc
# @File : convolutional.py
# 卷积
import os

from mnist import model

import tensorflow as tf

from mnist import input_data

data = input_data.read_data_sets('MNIST_data', one_hot=True)

# model

with tf.variable_scope('convolutional'):
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    keep_prob = tf.placeholder(tf.float32)
    y, variables = model.convolutional(x, keep_prob)

# train
y_ = tf.placeholder(tf.float32, [None, 10], name='y')

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # 随机梯度下降的方式

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # 判断参数是否相等

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(variables)

with tf.Session() as sess:
    merged_summary_op = tf.summary.merge_all()  # 合并参数，操作符
    summary_writer = tf.summary.FileWriter('/tmp/mnist_log/1', sess.graph)  # 将参数的路径、输入输出图放到哪里
    summary_writer.add_graph(sess.graph)  # 把图加进来
    sess.run(tf.global_variables_initializer())
# 1.在GPU上运行可能会稍微快一些比在CPU上。
    for i in range(20000):  # 对于这样的卷积训练一般要做10000-20000次的循环
        batch = data.train.next_batch(50)  # 定义batch的大小
        if i % 100 == 0:  # 每隔100次准确率做一次打印
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))  # 打印
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print(sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels, keep_prob: 1.0}))
# 保存
    path = saver.save(
        sess, os.path.join(os.path.dirname(__file__), 'data', 'convolutional.ckpt'),
        write_meta_graph=False, write_state=False
    )
    print('Saved:', path)