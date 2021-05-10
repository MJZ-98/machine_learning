# import numpy as np
# import pandas as pd
#
# data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
#         'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
#         'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
#         'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}
# labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
# df = pd.DataFrame(data=data, index=labels)
# # A:
# print(df.info())
# # 值和类型
# print(df.describe())
# print()
# # B:
# print(df.head(3))
# print()
# # C:
# print(df[['animal', 'age']])
# print()
# # D:
# print(df.iloc[[3, 4, 8]][['animal', 'age']])
# print()
# # E:
# print(df[df['age'] > 3])
# print()
# # F:
# print(df[pd.isna(df['age'])])


# import pandas as pd
# import numpy as np
#
# df = pd.DataFrame({'x': [1.2, np.nan, 3, 4], 'y': ['a', 'b', 'c', 'd']})
# # a:
# print(df == df)
# print()
# # b:
# print(df.isnull().sum())
# print()
# # c:
# for i in list(df.columns[df.isnull().sum() > 0]):
#     mean = df[i].mean()
#     df[i].fillna(mean, inplace=True)
# print(df)

# import numpy as np
# n = np.ones((10, 10))
# n[1:-1, 1:-1] = 0
# print(n)

# import numpy as np
# n = np.dot(np.random.rand(5, 3), np.random.rand(3, 2))
# print(n)

"""Flappy, game inspired by Flappy Bird.

Exercises

1. Keep score.
2. Vary the speed.
3. Vary the size of the balls.
4. Allow the bird to move f  orward and back.

"""

from random import *
from turtle import *
from freegames import vector

bird = vector(0, 0)
balls = []

def tap(x, y):
    "Move bird up in response to screen tap."
    up = vector(0, 50)#跃进
    bird.move(up)

def inside(point):
    "Return True if point on screen."
    return -200 < point.x < 200 and -200 < point.y < 200

def draw(alive):
    "Draw screen objects."
    clear()

    goto(bird.x, bird.y)

    if alive:
        dot(10, 'green')
    else:
        dot(10, 'red')

    for ball in balls:
        goto(ball.x, ball.y)
        dot(20, 'black')

    update()

def move():
    "Update object positions."
    bird.y -= 5

    for ball in balls:
        ball.x -= 3

    if randrange(10) == 0:#难度
        y = randrange(-199, 199)
        ball = vector(199, y)
        balls.append(ball)

    while len(balls) > 0 and not inside(balls[0]):
        balls.pop(0)

    if not inside(bird):
        draw(False)
        return

    for ball in balls:
        if abs(ball - bird) < 15:
            draw(False)
            return

    draw(True)
    ontimer(move, 50)

setup(420, 420, 370, 0)

hideturtle()
up()
tracer(False)
onscreenclick(tap)
move()
done()