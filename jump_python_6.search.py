# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 17:02:07 2019

@author: Administrator
"""

'''
06-6 하위 디렉터리 검색
특정 디렉터리부터 시작해서 그 하위 모든 파일 중 파이썬 파일(*.py)만 출력해 주는 
프로그램을 만들려면 어떻게 해야 할까?

#####

1.기본 골격을 설계한다. 함수정의(input/output) 
def search(dirname):
    print(dirname)
 
search("c:/")

2.로직함수를 사용한다
os.listdir(dirname) : filename 을 출력한다.
os.path.join(dirname, filename) : path+filename을 출력한다.
 for문을 사용해 쭈욱 읽어들여서 출력함

os.path.splitext(full_filename) : 확장자를 기준으로 파일을 나눔
 -1 : 확장자
 -2 : filename

#####

※ 재귀 호출이란 자기 자신을 다시 호출하는 프로그래밍 기법이다. 이 코드에서 보면 
search 함수에서 다시 자기 자신인 search 함수를 호출하는 것이 바로 재귀 호출이다.

#####

코딩도장 : http://codingdojang.com

'''

import os
def search(dirname):
    try:
        filenames = os.listdir(dirname)
        for filename in filenames :
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                search(full_filename)
            else :
                ext = os.path.splitext(full_filename)[-1]
                if ext == '.py':
                    print(full_filename, ext)
    except PermissionError:  #Error가 발생하더라도 지속 수행하도록
        pass
    
search("d:/개인/python")

for (path, dir, files) in os.walk("d:/개인/"):
    for filename in files:
        ext = os.path.splitext(filename)[1]
        if ext == '.py':
            print("%s/%s" % (path, filename))