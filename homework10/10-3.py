import numpy as np
import math


def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))


lrate = 0.5
INDIM = 26
H1DIM = 13
OUTDIM = 3

x = np.array([
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # T
          0.0, 0.0, 1.0, 0.0, 0.0,
          0.0, 0.0, 1.0, 0.0, 0.0,
          0.0, 0.0, 1.0, 0.0, 0.0,
          0.0, 0.0, 1.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # C
          1.0, 0.0, 0.0, 0.0, 0.0,
          1.0, 0.0, 0.0, 0.0, 0.0,
          1.0, 0.0, 0.0, 0.0, 0.0,
          1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # E
          1.0, 0.0, 0.0, 0.0, 0.0,
          1.0, 1.0, 1.0, 1.0, 1.0,
          1.0, 0.0, 0.0, 0.0, 0.0,
          1.0, 1.0, 1.0, 1.0, 1.0],
])
t = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])

w1 = np.zeros([H1DIM, INDIM])
for i in range(H1DIM):
    for j in range(INDIM):
        w1[i][j] = np.random.rand()
    pass
w2 = np.zeros([OUTDIM,H1DIM])
for i in range(OUTDIM):
    for j in range(H1DIM):
        w2[i][j] = np.random.rand()

y1 = np.zeros(H1DIM)
y2 = np.zeros(OUTDIM)

d1 = np.zeros(H1DIM)
d2 = np.zeros(OUTDIM)

for epoch in range(10000):
    if epoch % 100 == 0:
        print(y2)
    for p in range(3):
        # feed forwarding
        # LAYER-1 (Hidden Layer)
        for i in range(H1DIM):
            out = 0.0
            for j in range(INDIM):
                out += w1[i][j] * x[p][j]
            y1[i] = sigmoid(out)
        # LAYER-2 (OUTPUT Layer)
        for i in range(OUTDIM):
            out = 0.0
            for j in range(H1DIM):
                out += w2[i][j] * y1[j]
            y2[i] = sigmoid(out)
        # Back Propagation
        # delta(error) for Layer-2 (Output Layer)
        for i in range(OUTDIM):
            d2[i] = (t[p][i] - y2[i]) * (y2[i]*(1 - y2[i]))
        # delta(error) for Layer-1 (Hidden Layer)
        for i in range(H1DIM):
            for j in range(OUTDIM):
                d1[i] = d2[j] * w2[j][i] * (y1[i]*(1 - y1[i]))
        
        # Weight Adjustment for Layer-2 (Output Layer)
        for i in range(OUTDIM):
            for j in range(H1DIM):
                w2[i][j] += lrate * d2[i] * y1[j]
        
        # Weight Adjustment for Layer-1 (Hidden Layer)
        for i in range(H1DIM):
            for j in range(INDIM):
                w1[i][j] += lrate * d1[i] * x[p][j]
        

testcase = np.array([
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # R
          1.0, 0.0, 0.0, 0.0, 1.0,
          1.0, 1.0, 1.0, 1.0, 1.0,
          1.0, 0.0, 0.0, 1.0, 0.0,
          1.0, 0.0, 0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, # W
          1.0, 0.0, 1.0, 0.0, 1.0,
          1.0, 0.0, 1.0, 0.0, 1.0,
          1.0, 0.0, 1.0, 0.0, 1.0,
          0.0, 1.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, # I
          0.0, 0.0, 1.0, 0.0, 0.0,
          0.0, 0.0, 1.0, 0.0, 0.0,
          0.0, 0.0, 1.0, 0.0, 0.0,
          0.0, 1.0, 1.0, 1.0, 0.0],
])

y1 = np.zeros(H1DIM)
y2 = np.zeros(OUTDIM)

for p in range(3):
    # feed forwarding
    # LAYER-1 (Hidden Layer)
    for i in range(H1DIM):
        out = 0.0
        for j in range(INDIM):
            out += w1[i][j] * testcase[p][j]
        y1[i] = sigmoid(out)
    # LAYER-2 (OUTPUT Layer)
    for i in range(OUTDIM):
        out = 0.0
        for j in range(H1DIM):
            out += w2[i][j] * y1[j]
        y2[i] = sigmoid(out)
    
    for i in range(OUTDIM):
        y2[i] = round(y2[i],4)
    print(t[p], y2)
