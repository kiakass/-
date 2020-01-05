# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 13:57:41 2019

@author: Administrator
"""


# filter
'''
첫번째 인수로 함수 이름을, 두번째 인수로 자료형을 넣는다.
반환값이 참인것을 묶어서 돌려보냄
'''
def positive(l):
    result = []
    for i in l:
        if i > 0:
            result.append(i)
    return result

print(positive([1,-3,2,0,-5,6]))

def positive(x):
    return x > 0

list(filter(positive, [1,-3,2,0,-5,6]))

list(filter(lambda x: x > 0, [1, -3, 2, 0, -5, 6]))

#rnadom, time
import random,time
for i in range(1000):
    print(random.randint(1,1000))
    time.sleep(1)
    
import random
def random_pop(data):
    number = random.randint(0, len(data)-1)
    return data.pop(number)

if __name__ == "__main__":
    data = [1, 2, 3, 4, 5]
    while data: 
        print(random_pop(data))