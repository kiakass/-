# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 08:34:15 2019

@author: Administrator
"""

#05-1 클래스

"""
class Cookie:
    pass
    
a = Cookie()

a: 객체, a 는 Cookie의 인스턴스
객체(object)는 저장공간에서 할당되어 값을 가지거나 식별자에 의해 참조되는 공간
객체는 클래스의 인스턴스임

인스턴스 : 클래스로 만든 객체
메서드 : 클래스안에 구현된 함수, 파이썬 메서드의 첫번째 매개변수 이름은 관례적으로
         self 를 씀(자신을 전달하기 때문..)
"""

"""
클래스 구조 만들기
"""
class FourCal:
    pass
a = FourCal()  #a가 FourCal의 객체임
type(a)
'''
>>> a = FourCal()
>>> type(a)
<class '__main__.FourCal'>
'''
''' 
객체에 숫자 지정할 수 있게 만들기
self : 객체 전달 ex) a.setdata(4,2) -> a=self
'''
class FourCal:
    def setdata(self, first, second):  #메서드 : 클래스안에 구현된 함수
        self.first = first
        self.second = second
    def add(self):
        return self.first + self.second
    def sub(self):
        return self.first - self.second
    def mul(self):
        return self.first * self.second
    def div(self):
        return self.first / self.second        

a = FourCal()
a.setdata(4,2)  # or FourCal.setdata(a,4,2)
print(a.first)
print(a.second)

print(a.add())
a = FourCal()
a.setdata(4,2)  # or FourCal.setdata(a,4,2)
print(a.mul())
print(a.div())
print(a.sub())

''' 
생성자(Consructor) : 객체가 생성되는 시점에 자동 호출 __init__
'''
class FourCal:
    def __init__(self, first, second):  #메서드 : 클래스안에 구현된 함수
        self.first = first
        self.second = second
    def setdata(self, first, second):  #메서드 : 클래스안에 구현된 함수
        self.first = first
        self.second = second
    def add(self):
        return self.first + self.second
    def sub(self):
        return self.first - self.second
    def mul(self):
        return self.first * self.second
    def div(self):
        return self.first / self.second

a = FourCal(3,4)
print(a.mul())

'''
클래스의 상속 : 기존클래스를 변경하지 않고 기능을 추가하거나, 기존기능을 변경
'''
class MoreFourCal(FourCal):
    pass

a= MoreFourCal(4,2)
print(a.add())
print(a.mul())
print(a.div())
print(a.sub())

class MoreFourCal(FourCal):
    def pow(self):
        return self.first ** self.second
a= MoreFourCal(4,2)
print(a.pow())

'''
매서드 오버라이딩 : 상속받은 클래스에서 동일한 이름의 매서드 작성(덮어쓰기)
'''
class SafeFourCal(FourCal):
    def div(self):
        if self.second == 0 : return 0
        else : return self.first / self.second
a = SafeFourCal(4,0)
print(a.div())


'''
클래스 변수: 클래스로 만든 모든 객체에 공유
'''
class Family:
    lastname = '김'

a = Family()
b = Family()

print(Family.lastname)
print(a.lastname)
print(b.lastname)
id(Family.lastname)
id(a.lastname)
id(b.lastname)
