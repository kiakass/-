# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:40:51 2019

@author: Administrator
"""

'''
1.패키지란 무엇인가?
2.패키지 만들기
3.패키지 안의 함수 실행하기
4.__init__.py 의 용도
5.relative 패키지

패키지(Packages)는 도트(.)를 사용하여 파이썬 모듈을 계층적(디렉터리 구조)으로 
관리할 수 있게 해준다.
패키지는 디렉터리와 모듈로 이루어진다.

D:/개인/python/game/__init__.py
D:/개인/python/game/sound/__init__.py
D:/개인/python/game/sound/echo.py
D:/개인/python/game/graphic/__init__.py
D:/개인/python/game/graphic/render.py
'''

'''
패키지 안의 함수 실행하기
'''
import game.sound.echo
game.sound.echo.echo_test()

from game.sound.echo import echo_test # 매서드를 import 하는 방법
echo_test()

import game
game.sound.echo.echo_test()      # 해당 directory 의 모듈만 import 됨
import game.sound.echo.echo_test # 매서드를 import 할수 없음

'''
__init__.py : 해당디렉터리가 패키지임을 인식
__all__ = ['echo']  # '*' 일 경우 echo.py 모듈을 import 해라
'''
from game.sound import *
echo.echo_test()

'''
relative 패키지
'''
