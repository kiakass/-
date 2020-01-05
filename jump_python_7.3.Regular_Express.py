# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 08:32:21 2019

@author: Administrator
"""

import re
p = re.compile('[a-z]+')
m = p.match('python')
# 축약된 형태
m = re.match('[a-z]+', 'python')

# MULTILINE
p = re.compile("^python\s\w+", re.MULTILINE)

p = re.compile('Life$', re.M)
print(p.search('My Life'))

# Raw String 'r'
'''
메타문자를 사용할시 백스페이스가 아닌 단어 구분자
p = re.compile(r'\bclass\b)
'''
p = re.compile(r'\bclass\b')
print(p.search('no class at all'))  
p = re.compile('class')
print(p.search('the declassified algorithm'))

'''
\B
'''
p = re.compile(r'\Bclass\B')
print(p.search('the declassified algorithm'))

'''
그루핑 ( )
'''
p = re.compile('(ABC)+')
m = p.search('ABCABCABC OK?')
print(m)
print(m.group())

p = re.compile(r"\w+\s+\d\d\d[-]\d+[-]\d+")

p = re.compile(r"(\w+)\s+(\d+)[-]((\d+)[-](\d+))")
m = p.search("park 010-1234-1235")
print(m)
print(m.group())
print('#0 '+m.group(0))
print('#1 '+m.group(1))
print('#2 '+m.group(2))
print('#3 '+m.group(3))
print('#4 '+m.group(4))
print('#5 '+m.group(5))     

p = re.compile(r'((\b\w+)\s+)\2')

p.search('Paris in in the the spring').group()

p = re.compile(".+(?=:)")
p.search("http://google.com").group()

###모르겠음.....
