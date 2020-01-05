# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 15:16:49 2019

@author: Administrator

참조 Site : 
https://gist.github.com/narphorium/d06b7ed234287e319f18

"""

import tensorflow as tf
import pandas as pd   # 데이터조작
import seaborn as sns # 시각화패키지
import matplotlib.pyplot as plt
import numpy as np
#sess = tf.Session()
#tf.global_variables_initializer()

num_points = 10
vectors_set,vectors1_set,vectors2_set,vectors3_set =[], [], [], []

for i in range(num_points):
    if np.random.random() > 0.5:  # random.random 0~1 사이에 값을 랜덤하게 돌려줌
        vectors_set.append([np.random.normal(0.0,0.9),
                            np.random.normal(0.0,0.9)])
    else:
        vectors_set.append([np.random.normal(3.0,0.5),
                            np.random.normal(1.0,0.5)])
print(len(vectors1_set))
df = pd.DataFrame({"x": [v[0] for v in vectors_set],
                   "y": [v[1] for v in vectors_set]})
sns.lmplot("x", "y", data=df, fit_reg=False, size=6)

plt.show()  

vectors = tf.constant(vectors_set)
k = 2
centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors),[0,0],[k,-1]))

vectors.get_shape()
centroids.get_shape()
#'''
print(sess.run(centroids)) # 변수는 initialize해서 값을 채워야함 print됨
print(sess.run([vectors,centroids]))
#print(sess.run(tf.Variable(tf.slice(tf.random_shuffle(vectors),[0,0],[k,-1]))))
#'''
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroids = tf.expand_dims(centroids, 1)
#'''
print(expanded_vectors.get_shape(),expanded_centroids.get_shape())
print(expanded_centroids)
#'''
distance = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors,expanded_centroids)),2)
assignments = tf.argmin(distance,0)
assignments.get_shape()
#'''
print(sess.run(expanded_vectors))
print(sess.run(expanded_centroids))
print(sess.run(tf.subtract(expanded_vectors, expanded_centroids)))
t = tf.square(tf.subtract(expanded_vectors,expanded_centroids))
print(sess.run(t),'\n''\n',sess.run(tf.reduce_sum(t,2)))
print(sess.run(tf.square(tf.subtract(expanded_vectors,expanded_centroids))))
print(sess.run(tf.round(distance)))
tf.subtract(expanded_vectors, expanded_centroids).shape
print(sess.run(assignments))
#'''
means = tf.concat([tf.reduce_mean
                   ( tf.gather(vectors,
                               tf.reshape(tf.where(tf.equal(assignments, c)), 
                                          [1,-1])
                               ),
                               reduction_indices=[1]
                   ) for c in range(k) # reduce_mean, 리스트 내포
                  ], 0 #concat 
                 )
#'''
c = 0
#print(sess.run([vectors]),'\n',sess.run(centroids))
print(sess.run(tf.equal(assignments, c)))
print(sess.run(tf.where(tf.equal(assignments, c))))
print(sess.run(tf.reshape(tf.where(tf.equal(assignments, c)), [1,-1])))
near_point = tf.gather(vectors,tf.reshape(tf.where(tf.equal(assignments, c)), [1,-1]))
print(sess.run(round(near_point)),near_point.shape)
print(sess.run(tf.reduce_mean(near_point, reduction_indices=[1] )))

print(sess.run([centroids]),'\n''\n',sess.run([means]))
#'''
#print(sess.run(means))

update_centroids = tf.assign(centroids, means)
'''
print(sess.run([update_centroids]))
#'''
init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)

for step in range(100):
    _, centroid_values, assignment_values = sess.run([update_centroids, 
                                                      centroids, assignments])
'''    
print('centroids')
#print(sess.run(centroids))
print(sess.run([assignments]), assignment_values)
print(vectors_set)
print(vectors_set[3][0])
#'''
data = {"x": [], "y": [], "cluster": []}

for i in range(len(assignment_values)):
    data["x"].append(vectors_set[i][0])
    data["y"].append(vectors_set[i][1])
    data["cluster"].append(assignment_values[i])


df = pd.DataFrame(data)
sns.lmplot("x", "y", data=df, fit_reg=False, size=7, hue="cluster", legend=False)
plt.show()
