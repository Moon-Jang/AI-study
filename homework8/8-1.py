import numpy as np
import random


def act_tlu(out):
    if out > 0.0:
        return 1.0
    return 0.0


def act_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def get_error(target, b):
    return target - b


x = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
t = np.array([0.0, 1.0, 1.0, 0.0])

lrate = 0.1

w = np.zeros(3)

w[0] = random.random()
w[1] = random.random()
w[2] = random.random()

for epoch in range(1000):
    b = []
    for i in range(4):
        out = w[0] * x[i][0] + w[1] * x[i][1] + w[2] * x[i][2]
        b.append(act_sigmoid(out))
        error = get_error(t[i], b[i])
        for j in range(3):
            w[j] = w[j] + lrate * error * x[i][j]

print("XOR")
for i in range(len(x)):
    print("입력:", x[i][1], x[i][2], " 출력 : ", b[i])