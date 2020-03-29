# -*- coding: utf-8 -*-
# @Time : 2020/3/29 17:57
# @Author : fc
# @File : main.py
import numpy as np

import tensorflow as tf

from flask import Flask, jsonify, render_template, request

from mnist import model

x = tf.placeholder('float', [None, 784])  # 声明输入

sess = tf.Session()  # 定义一个Session

# 拿线性回归模型，ckpt

with tf.variable_scope('regression'):
    y1, variables = model.regression(x)

saver = tf.train.Saver(variables)

saver.restore(sess, 'mnist/data/regression.ckpt')  # 通过restore方法把模型文件拿出来

# 拿卷积模型和线性同样的方法

with tf.variable_scope('convolutional'):
    keep_prob = tf.placeholder('float')

    y2, variables = model.convolutional(x, keep_prob)

saver = tf.train.Saver(variables)

saver.restore(sess, 'mnist/data/convolutional.ckpt')


def regression(input):
    return sess.run(y1, feed_dict={x: input}).flatten().tolist()  # 转换成list


def convolutional(input):
    return sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()


# 以上步骤之后，两个模型的输入以及如何把数据喂进来以及如何跑这个模型已经完成


# 做接口，用Flask

app = Flask(__name__)


@app.route('/', methods=['get'])  # 定义一个注解，路由，表示前端传进来之后应该用哪个接口
def helloworld():
    return render_template("index.html")
@app.route('/api/mnist', methods=['post'])  # 定义一个注解，路由，表示前端传进来之后应该用哪个接口
def mnist():
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)  # 做一个数组的换算，我们模型定义的形状就是1*784的形状

    output1 = regression(input)

    output2 = convolutional(input)

    return jsonify(results=[output1, output2])  # 把结果封装一下


# 定义一个方法来启动它

if __name__ == '__main__':
    app.debug = True

    app.run(host='0.0.0.0', port=8000)

