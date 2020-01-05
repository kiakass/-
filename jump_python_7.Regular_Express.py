# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:58:57 2019

@author: Administrator
"""

'''
정규 표현식(Regular Expressions)은 복잡한 문자열을 처리할 때 사용하는 기법으로, 
파이썬만의 고유 문법이 아니라 문자열을 처리하는 모든 곳에서 사용한다. 정규 표현식을
배우는 것은 파이썬을 배우는 것과는 또 다른 영역의 과제이다.
'''


data = """
park 800905-1049118
kim  700905-1059119
"""

result = []

for line in data.split("\n"):
    print(line)
    word_result = []
    for word in line.split(" "):
        print(word)
        if len(word) == 14 and word[:6].isdigit() and word[7:].isdigit():
            word = word[:6] + "-" + "*"*7
        word_result.append(word)
        print(word_result)
    result.append(" ".join(word_result))
    print(result)
    
print("\n".join(result))

import re 

data = """
park 800905-1049118
kim  700905-1059119
"""

pat = re.compile("(\d{6})[-]\d{7}")
print(pat.sub("\g<1>-*******", data))

# 07-2 정규표현식 기초,메타문자

'''
정규 표현식(Regular Expressions)은 복잡한 문자열을 처리할 때 사용하는 기법
메타문자 : 원래 그 문자가 가진 뜻이 아닌 특별한 용도로 사용하는 문자
. ^ $ * + ? { } [ ] \ | ( )

[] : 문자 클래스 "[ ] 사이의 문자들과 매치"
[a-zA-Z] : 알파벳 모두
[0-9] : 숫자
[^0-9]: 숫자가 아닌 문자

######

\d - 숫자와 매치, [0-9]와 동일한 표현식이다.
\D - 숫자가 아닌 것과 매치, [^0-9]와 동일한 표현식이다.
\s - whitespace 문자와 매치, [ \t\n\r\f\v]와 동일한 표현식이다. 맨 앞의 빈 칸은 
     공백문자(space)를 의미한다.
\S - whitespace 문자가 아닌 것과 매치, [^ \t\n\r\f\v]와 동일한 표현식이다.
\w - 문자+숫자(alphanumeric)와 매치, [a-zA-Z0-9_]와 동일한 표현식이다.
\W - 문자+숫자(alphanumeric)가 아닌 문자와 매치, [^a-zA-Z0-9_]와 동일한 표현식이다.

######
'''
