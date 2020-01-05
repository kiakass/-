import tensorflow as tf
import pandas as pd   # 데이터조작
import seaborn as sns # 시각화패키지
import matplotlib.pyplot as plt
import numpy as np
sess = tf.Session()
tf.global_variables_initializer()

num_points = 1000
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

expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroids = tf.expand_dims(centroids, 1)

# Cost 함수
distance = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors,expanded_centroids)),2)
assignments = tf.argmin(distance,0)
assignments.get_shape()

means = tf.concat([tf.reduce_mean
                   ( tf.gather(vectors,
                               tf.reshape(tf.where(tf.equal(assignments, c)), 
                                          [1,-1])
                               ),
                               reduction_indices=[1]
                   ) for c in range(k) # reduce_mean, 리스트 내포
                  ], 0 #concat 
                 )


update_centroids = tf.assign(centroids, means)

init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)

'''
값을 찍어보기 위한 출력문
# distance 구하기
print(sess.run([vectors]),'\n\n',sess.run(centroids))
print(sess.run([expanded_vectors]),'\n\n',sess.run(expanded_centroids))
print(sess.run(tf.subtract(expanded_vectors, expanded_centroids)))
t = tf.square(tf.subtract(expanded_vectors,expanded_centroids))
print(sess.run(t),'\n''\n',sess.run(tf.reduce_sum(t,2)))

print(sess.run(tf.round(distance)))
tf.subtract(expanded_vectors, expanded_centroids).shape
print(sess.run(assignments))

# mean 구하기
c = 0
print(sess.run([vectors]),'\n',sess.run(centroids))
print('equal : ','\n',sess.run(tf.equal(assignments, c)))
print(sess.run(tf.where(tf.equal(assignments, c))))
print(sess.run(tf.reshape(tf.where(tf.equal(assignments, c)), [1,-1])))
near_point = tf.gather(vectors,tf.reshape(tf.where(tf.equal(assignments, c)), [1,-1]))
print(sess.run((near_point)),near_point.shape)
print(sess.run(tf.reduce_mean(near_point, reduction_indices=[1] )))

print(sess.run([centroids]),'\n''\n',sess.run([means]))
#'''


update_centroids = tf.assign(centroids, means)
print('----------------------')


for step in range(10):
    _, centroid_values, assignment_values= \
       sess.run([update_centroids, centroids, assignments])
#    print(step,'\n', centroid_values,'\n\n', assignment_values)

'''
# 값을 찍어 보기
for step in range(10):
    _, centroid_values, assignment_values, mean_val, dist_val = \
       sess.run([update_centroids, centroids, assignments, means, distance])
    print(step, 'distance: ','\n',dist_val,'\n\n',centroid_values,'\n\n', mean_val, '\n\n', \
          assignment_values)
    print('----------------------')
'''

data = {"x": [], "y": [], "cluster": []}

for i in range(len(assignment_values)):
    data["x"].append(vectors_set[i][0])
    data["y"].append(vectors_set[i][1])
    data["cluster"].append(assignment_values[i])


df = pd.DataFrame(data)
sns.lmplot("x", "y", data=df, fit_reg=False, size=7, hue="cluster", legend=False)
plt.show()
