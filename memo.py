# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:36:13 2019

@author: Administrator
"""

# 메모장 만들기

import sys

option = sys.argv[1]

if option == '-a':
    memo = sys.argv[2]
    f = open('D:/개인/python/memo.txt','a')
    f.write(memo)
    f.write('\n')
    f.close()
elif option == '-v':
    f = open('D:/개인/python/memo.txt','r')
    print(f.read())
    f.close()