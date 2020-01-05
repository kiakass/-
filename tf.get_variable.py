# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 22:58:37 2020

@author: junhk
"""
import tensorflow as tf

a1 = tf.Variable(1.)
a2 = tf.Variable(2.)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    print("\t a1 name: ", a1.name)
    print("\t a2 name: ", a2.name)
    
    a1_1, a2_1 = sess.run(["Variable_2:0", "Variable_3:0"])
    #a1_1, a2_1 = sess.run([a1, a2])
    # a1, a2
    
    print("\t a1 value: %f" % a1_1)
    print("\t a2 value: %f" % a2_1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    a3 = tf.Variable(3., name="a3")
    a4 = tf.Variable(4., name="a4")
    a5 = tf.get_variable("a5", 1, initializer=tf.constant_initializer(5.))
    a6 = tf.get_variable("a6", 1, initializer=tf.constant_initializer(6.))
    
    print("\t a3 name: ", a3.name)
    print("\t a4 name: ", a4.name)
    print("\t a5 name: ", a5.name)
    print("\t a6 name: ", a6.name)
    
    
with tf.variable_scope("DEF", reuse=True):
    a5_ = tf.get_variable("a5")
    a6_ = tf.get_variable("a6")    
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    print("\t a5 name: ", a5_.name)
    print("\t a6 name: ", a6_.name)