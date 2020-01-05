# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 07:40:35 2019

@author: Administrator
"""

#
'''
06-5 탭을 개의 공백으로 바꾸기

a.txt 파일을 읽어들여
tab을 ''으로 대체한후
b.txt 에 cop함
'''
import sys

#before = sys.argv[1]
f = open('a.txt','r')
body = f.read()
body = body.replace('\t',' '*4)
f.close()

#after = sys.argv[2]
f = open('b.txt','w')
f.write(body)
f.close()

with open('b.txt','r') as f:
    print(f.read())