# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 16:03:07 2019

@author: Administrator
"""
'''
오류는 어떤 때 발생하는가?
오류 예외 처리 기법
try, except문
try .. finally
여러개의 오류처리하기
오류 회피하기
오류 일부러 발생시키기
예외 만들기
'''

'''
try .. except

except 발생 오류: #발생한 오류만을 포함
except 발생 오류 as 오류 메시지 변수: #발생한 오류와 오류메시지포함
'''


try:
    4/0
except ZeroDivisionError as e:
    print(e)

'''
try finally : 무조건 수행됨
'''
f = open('foo.txt', 'w')
try:
    #무언가를 수행한다.
finally:
    f.close()

'''
여러개 오류 처리하기
'''
try:
    a = [1,2]
    print(a[3])
    4/0
except (ZeroDivisionError, IndexError) as e:
    print(e)
    
'''
오류 회피하기
'''
try:
    f = open('나없는파일','r')
except FileNotFoundError:
    pass

'''
오류 일부러 발생시키기
'''
class Bird:
    def fly(self):
        raise NotImplementedError
        
''' 클래스를 상속받고 fly를 구현하지 않으면 에러가 발생함 '''
class Eagle(Bird):
    pass
eagle = Eagle()
eagle.fly()

class Eagle(Bird):
    def fly(self):
        print("very fast")
eagle = Eagle()
eagle.fly()

'''
예외 만들기
'''
class MyError(Exception):
    pass

def say_nick(nick):
    if nick == '바보':
        raise MyError()
    print(nick)
    
say_nick('천사')
say_nick('바보')

try:
    say_nick('천사')
    say_nick('바보')
except MyError:
    print('허용되지 않는 별명입니다.')

'''
 __str__ 메서드는 print(e)처럼 오류 메시지를 
 print문으로 출력할 경우에 호출되는 메서드
'''    

class MyError(Exception):
    def __str__(self):
        return '허용되지 않는 별명입니다.'
try:
    say_nick("천사")
    say_nick("바보")
except MyError as e:
    print(e)