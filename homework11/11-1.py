import numpy as np
import math
import copy

def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))


def shuffle(arr1,arr2):
    for i in range(len(arr1)):
        randIdx = np.random.randint(i,len(arr1))
        swap(arr1,i,randIdx)
        swap(arr2,i,randIdx)


def swap(arr,i,j):
    temp = copy.deepcopy(arr[i])
    arr[i] = copy.deepcopy(arr[j])
    arr[j] = temp


lrate = 0.1
INDIM = 50
H1DIM = 25
H2DIM = 12
OUTDIM = 3

x = np.array([
    [1.0,1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # T-1
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    [1.0,0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # C-1
         0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0,1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # E-1
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0,1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # T-2
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    [1.0,1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # C-2
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0,0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # E-2
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0,0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,  # T-3
         0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [1.0,0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,  # C-3
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
         1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
    [1.0,0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,  # E-3
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
])
t = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])

w1 = np.zeros([H1DIM, INDIM])
for i in range(H1DIM):
    for j in range(INDIM):
        w1[i][j] = np.random.rand() / 10.0
    pass
w2 = np.zeros([H2DIM, H1DIM])
for i in range(H2DIM):
    for j in range(H1DIM):
        w2[i][j] = np.random.rand() / 10.0
w3 = np.zeros([OUTDIM, H2DIM])
for i in range(OUTDIM):
    for j in range(H2DIM):
        w3[i][j] = np.random.rand() / 10.0

y1 = np.zeros(H1DIM)
y2 = np.zeros(H2DIM)
y3 = np.zeros(OUTDIM)

d1 = np.zeros(H1DIM)
d2 = np.zeros(H2DIM)
d3 = np.zeros(OUTDIM)

for epoch in range(10000):
    if (epoch % 100) == 0:
        print("epoch", epoch)
    for p in range(len(x)):
        # feed forwarding
        # LAYER-1 (Hidden Layer)
        for i in range(H1DIM):
            out = 0.0
            for j in range(INDIM):
                out += w1[i][j] * x[p][j]
            y1[i] = sigmoid(out)
        # LAYER-2 (OUTPUT Layer)
        for i in range(H2DIM):
            out = 0.0
            for j in range(H1DIM):
                out += w2[i][j] * y1[j]
            y2[i] = sigmoid(out)
        # LAYER-3 (OUTPUT Layer)
        for i in range(OUTDIM):
            out = 0.0
            for j in range(H2DIM):
                out += w3[i][j] * y2[j]
            y3[i] = sigmoid(out)
        # Back Propagation
        # delta(error) for Layer-3 (Output Layer)
        for i in range(OUTDIM):
            d3[i] = (t[p][i] - y3[i]) * (y3[i] * (1 - y3[i]))
        # Back Propagation
        # delta(error) for Layer-2 (Output Layer)
        for i in range(H2DIM):
            for j in range(OUTDIM):
                d2[i] = d3[j] * w3[j][i] * (y2[i] * (1 - y2[i]))
        # delta(error) for Layer-1 (Hidden Layer)
        for i in range(H1DIM):
            for j in range(H2DIM):
                d1[i] = d2[j] * w2[j][i] * (y1[i] * (1 - y1[i]))

        # Weight Adjustment for Layer-3 (Output Layer)
        for i in range(OUTDIM):
            for j in range(H2DIM):
                w3[i][j] += lrate * d3[i] * y2[j]
        # Weight Adjustment for Layer-2 (Output Layer)
        for i in range(H2DIM):
            for j in range(H1DIM):
                w2[i][j] += lrate * d2[i] * y1[j]

        # Weight Adjustment for Layer-1 (Hidden Layer)
        for i in range(H1DIM):
            for j in range(INDIM):
                w1[i][j] += lrate * d1[i] * x[p][j]

        if (epoch % 100) == 0:
            print(t[p], y3)
