# -*- coding: utf-8 -*-
# @Time : 2020/3/25 22:34
# @Author : fc
# @File : regression.py
import os
# import input_data
# data=input_data.read_data_sets('MNIST_data',one_hot=True)
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import datasets # 导入经典数据集加载模块
# 加载MNIST 数据集
(x, y), (x_test, y_test) = datasets.mnist.load_data()
print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:',
y_test)