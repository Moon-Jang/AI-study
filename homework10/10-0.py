import numpy as np
import math


def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))


lrate = 0.1
INDIM = 3
H1DIM = 3
OUTDIM = 1

x = np.array([
    [1.0, 0.1, 0.1],
    [1.0, 0.1, 1.0],
    [1.0, 1.0, 0.1],
    [1.0, 1.0, 1.0]
])
t = np.array([0.0, 1.0, 1.0, 0.0])

w1 = np.zeros([H1DIM, INDIM])
for i in range(H1DIM):
    for j in range(INDIM):
        w1[i][j] = np.random.randint(1, 10) / 10
    pass
w2 = np.zeros(H1DIM)
for i in range(H1DIM):
    w2[i] = np.random.randint(1, 10) / 10
    
y1 = np.zeros(H1DIM)
y2 = 0.0

d1 = np.zeros(H1DIM)
d2 = 0.0

for epoch in range (100000):
    if (epoch % 10) == 0:
        print("epoch",epoch)
    for p in range(4):
        # feed forwarding
        # LAYER-1 (Hidden Layer)
        for i in range(H1DIM):
            out = 0.0
            for j in range(INDIM):
                out += w1[i][j] * x[p][j]
            y1[i] = sigmoid(out)
        # LAYER-2 (OUTPUT Layer)
        out = 0.0
        for i in range(H1DIM):
            out += w2[i] * y1[i]
        y2 = sigmoid(out)
        
        # Back Propagation
        # delta(error) for Layer-2 (Output Layer)
        d2 = (t[p] - y2)
        # delta(error) for Layer-1 (Hidden Layer)
        for i in range(H1DIM):
            d1[i] = d2 * w2[i]
        
        # Weight Adjustment for Layer-2 (Output Layer)
        for i in range(H1DIM):
            w2[i] += lrate * d2 *(y2*(1-y2)) * y1[i]
        
        # Weight Adjustment for Layer-1 (Hidden Layer)
        for i in range(H1DIM):
            for j in range(INDIM):
                w1[i][j] += lrate * d1[i] * (y1[i]*(1-y1[i])) * x[p][j]

        if (epoch % 10) == 0:
            print(t[p],y2)