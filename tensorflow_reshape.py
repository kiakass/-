# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 16:28:18 2019

@author: Administrator
"""
import tensorflow as tf
import numpy as np
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
# np.array
t = np.array([0., 1., 2., 3., 4., 5., 6., 7.,8.,9.,10.,11.]).reshape(3,4)
print(t,t.shape)

t = np.array([0., 1., 2., 3., 4., 5., 6., 7.,8.,9.,10.,11.]).reshape(-1,2)
print(t,t.shape)

t = np.array([0., 1., 2., 3., 4., 5., 6., 7.,8.,9.,10.,11.]).reshape(-1,6)
print(t,t.shape)

t = np.array([0., 1., 2., 3., 4., 5., 6., 7.,8.,9.,10.,11.]).reshape(2,3,2)
print(t,t.shape)
t.reshape(-1)

print(t.reshape(-1))
print(t.reshape(12,-1))
print(t.reshape(-1,2,3))
print(t.reshape(-1,2,1,1,3))




t.reshape(-1,2)
q=np.expand_dims(t,0)
print(q,q.shape)
q=np.expand_dims(t,1)
print(q,q.shape)
print(q.reshape(-1,2))
print(q.reshape(-1))



tf.expand_dims(t,0).eval()


t = np.array([[1.,2.,3.,],[4.,5.,6.],[7.,8.,9.],[10.,11.,12.]])
# 정돈하여 프린트
print('array : \n',t)
print('dim :', t.ndim,'\nshape :',t.shape)#print(q).eval()

t = np.array([[1.,2.,3.,],[4.,5.,6.],[7.,8.,9.],[10.,11.,12.]])

# 정돈하여 프린트

print('array : \n',t)

print('dim :', t.ndim,'\nshape :',t.shape)

## reshape 을 활용해 원하는 형태로 데이터의 형태를 가공할 수 있음

x = np.array([[[1,1,1],[2,2,2]],[[3,3,3],[4,4,4]]])

print("X.shape: ", x.shape)

X.shape: (2, 2, 3) # rank 3, 요소 3


y = tf.reshape(x, shape=[-1,3]).eval(session=sess) # -1 = 4

print(y,'\n',y.shape)

﻿
# 요소수를 2개로 조정
y = tf.reshape(x, shape=[-1,2]).eval(session=sess)
print(y,'\n',y.shape)

#Rank 를 3차원:요소2, 2차원:요소2, 1차원:요소수3(=-1) (요소수12개=2*2*3) 로 조정
y = tf.reshape(x, shape=[-1,2,2]).eval(session=sess)
print(y,'\n',y.shape)


y = tf.reshape(x, shape=[-1,1]).eval(session=sess)
print(y,'\n',y.shape)

# Rank 1, 요소수 12
y = tf.reshape(x, shape=[-1]).eval(session=sess) # -1 = 12
print(y,'\n',y.shape)

﻿
﻿
﻿
﻿

﻿
﻿