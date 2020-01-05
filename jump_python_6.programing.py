# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 22:15:42 2019

@author: Administrator
"""

for i in range(2,10):
    print(" ")
    for j in range(1,10):
        print(i*j, end=" ")
        
        
def GuGu(n):
    result = []
    result.append(n*1)
    result.append(n*2)
    result.append(n*3)
    result.append(n*4)
    result.append(n*5)
    result.append(n*6)
    result.append(n*7)
    result.append(n*8)
    result.append(n*9)
    return result

print(GuGu(int(input())))

def GuGu(n):
    i = 0
    result = []
    while i < 10:
        i += 1
        result.append(n*i)
    return result
print(GuGu(int(input())))

'''
10 미만의 자연수에서 3과 5의 배수를 구하면 3, 5, 6, 9이다. 이들의 총합은 23이다.
1000 미만의 자연수에서 3의 배수와 5의 배수의 총합을 구하라.
'''
import time

def Addmul(o,p):
    i,j,k = 1,0,0
    while i < 1000:
        if i % o == 0 or i % p == 0:
            j += i
#            time.sleep(1)
        i += 1
        print("i,j,k %d %d" % (i,j))
    return j

print(Addmul(3,5))

result = 0
for i in range(1,1000):
    if i % 3 == 0 or i % 5 == 0:
        result += i
print(result)

