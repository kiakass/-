# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 14:42:50 2019

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
 
x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [2]]
x = np.asarray(x_data)
y = np.reshape(y_data,[-1])
#'''
print(x,'\n',y)
#'''
print(x[:,0])
print(x[:,1])

df = pd.DataFrame(dict(x=x[:,0], y=x[:,1], color=y))
sns.lmplot('x', 'y', data=df, hue='color', fit_reg=False)
plt.show()

import random
for i in range(0,30):
    x = random.randint(0,100)
    print(x)