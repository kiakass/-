# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 07:32:33 2019

@author: Administrator
"""
""" 여러개의 입력값을 받는 함수 만들기 """
def add_many(*args):
    result = 0
    for i in args:
        result += i
    return result


print(add_many(1,2,3,4,5,6,7,8,))

def add_mul(choice, *args):
    if choice == 'add' :
        result = 0
        for i in args:
            result += i
    elif choice == 'mul' :
        result = 1
        for i in args:
            result *= i
    return result

result = add_mul('add', 1,2,3,4,5)
print(result)

result = add_mul('mul', 1,2,3,4,5)
print(result)

""" 키워드 파라미터 kwargs : 값을 딕셔너리로 만들어서 출력 """
def print_kwargs(**kwargs):
    print(kwargs)

print_kwargs(a=1,name='foo', age=3)

def add_and_mul(a,b):
    return a+b, a*b

print(add_and_mul(3,4))
result1, result2 = add_and_mul(3,4)
print(result1, result2)

""" return : 함수빠져나가기 """
def say_nick(nick):
    if nick == '바보':
        return
    print('나의 별명은 %s 입니다.' % nick )

say_nick('야호')
say_nick('바보')

def say_myself(name, old, man=True): 
    print("나의 이름은 %s 입니다." % name) 
    print("나이는 %d살입니다." % old) 
    if man: 
        print("남자입니다.")
    else: 
        print("여자입니다.")
        
say_myself("박응용", 27)
say_myself("박응용", 27, True)
say_myself("박응선", 27, False)

#a = 1
def vartest(a):
    a = a +1
#    return a
vartest(a)
print(vartest(4))

def vartest(b):
    b = b + 1

vartest(3)
print(b)


""" 함수 안에서 함수 밖의 변수를 변경하는 방법 - return """
a = 1
def vartest(a):
    a += 1
    return a

a = vartest(a)
print(a)
""" 함수 안에서 함수 밖의 변수를 변경하는 방법 - 글로별 변수 """
a = 1
def vartest():
    global a
    a += 1
    
vartest()
print(a)

# 연습문제
''' Q3 '''
input1 = int(input("첫번째 숫자를 입력하세요:"))
input2 = int(input("두번째 숫자를 입력하세요:"))

total = input1 + input2
print("두수의 합은 %s 입니다." % total)

''' Q4 '''
print("".join(["you", "need", "python"]))
print("you", "need", "python")

''' Q5 '''
f1 = open("test.txt", 'w')
f1.write("Life is too short")
f1.close()
f2 = open("test.txt", 'r')
print(f2.read())
f2.close()

with open('test.txt', 'w') as f1:
    f1.write('Life is too short')

''' 
Q6: 사용자의 입력을 test.txt에 저장 
'''
a = input()
with open('test.txt', 'w') as f1:
    f1.write(a)
with open('test.txt', 'r') as f1:
    print(f1.read())
    

'''
Q7 : java 를 python으로 바꿔서 파일에 입력
'''
with open('test.txt','w') as f:
    f.write('Life is too short\n')
with open('test.txt','a') as f:
    f.write('you need java')
with open('test.txt','r') as f:
    body = f.read()
body = body.replace('java','python')
with open('test.txt','w') as f:
    f.write(body)
with open('test.txt', 'r') as f:
    print(f.read())