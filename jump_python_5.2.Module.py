# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 09:04:04 2019

@author: Administrator
"""

'''
모듈이란 함수나 변수 또는 클래스를 모아 놓은 파일이다. 
1.모듈 만들기
2.모듈 불러오기
3.if __name__ == "__main__": 의 의미
4.클래스나 변수 등을 포함한 모듈
5.다른 파일에서 모듈 불러오기
'''



'''
모듈 불러오기:
import 모듈이름 file.py 'py'를 뺀것 
'''
import mod1
print(mod1.add(3,5))

''' from 모듈이름 import 모듈함수 '''
from mod1 import add, sub
add(3,4)
sub(5,4)

'''
__name__ 변수 : 파이썬이 내부적으로 사용하는 특별한 변수
- 실행(>python mod1.py) 시 '__name__' 저장됨
- import 시 모듈이름값 mod1이 저장됨
'''
import mod1

'''
클래스나 변수 등을 포함한 모듈
'''
import mod2
print(mod2.PI) #변수
a = mod2.Math() #클래스
print(a.solv(2)) #매서드
print(mod2.add(mod2.PI, 4.4))

'''
다른 파일에서 모듈 불러오기 : sys.path.append("D:\개인\python")
'''
import sys
sys.path.append("D:\개인\python")