# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 14:07:24 2019

@author: Administrator
"""
import tensorflow as tf
import numpy as np
sess = tf.Session()
sess.run(tf.global_variables_initializer())


t1 = [[[1, 2], [2, 3]], [[4, 4], [5, 3]]]
t2 = [[[7, 4], [8, 4]], [[2, 10], [15, 11]]]
tf.concat([t1, t2], -1)
print(sess.run(tf.concat([t1, t2], 0)))
print(sess.run(tf.concat([t1, t2], 1)))
print(sess.run(tf.concat([t1, t2], 2)))


t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
print(sess.run(tf.concat([t1, t2], 0))) # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

print(sess.run(tf.concat([t1, t2], 1))) # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]﻿


t1 = [[[1, 2], [2, 3]], [[4, 4], [5, 3]]]
t2 = [[[7, 4], [8, 4]], [[2, 10], [15, 11]]]
print(sess.run(tf.concat([t1, t2], 0)))
print(sess.run(tf.concat([t1, t2], 1)))
print(sess.run(tf.concat([t1, t2], 2)))
