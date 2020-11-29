import numpy as np

print("5-1 프로젝션 값")
print("T\tC\tE")
print("5\t4\t5")
print("1\t1\t1")
print("1\t1\t5")
print("1\t1\t1")
print("1\t4\t5")
print("############################################################################")
print("5-2")
learn_data = [
    [1.0, 1.0, 1.0, 1.0, 1.0,  # T
     0.0, 0.0, 1.0, 0.0, 0.0,
     0.0, 0.0, 1.0, 0.0, 0.0,
     0.0, 0.0, 1.0, 0.0, 0.0,
     0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 1.0, 1.0,  # C
     1.0, 0.0, 1.0, 0.0, 0.0,
     1.0, 0.0, 0.0, 0.0, 0.0,
     1.0, 0.0, 0.0, 0.0, 0.0,
     0.1, 1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0,  # E
     1.0, 0.0, 0.0, 0.0, 0.0,
     1.0, 1.0, 1.0, 1.0, 1.0,
     1.0, 0.0, 0.0, 0.0, 0.0,
     1.0, 1.0, 1.0, 1.0, 1.0],
]


def act_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


patterns = ["T", "C", "E"]

for i in range(len(learn_data)):
    print("pattern:" + patterns[i])
    for j in range(5):
        row_str = ""
        for k in range(5):
            row_str += str(learn_data[i][(j * 5) + k]) + " "
        print(row_str)
    print()

print("############################################################################")
print("5-3")
w = np.zeros((3, 25))

for i in range(3):
    for j in range(25):
        w[i][j] = np.random.randint(1, 20) / 100.0

target = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
lrate = 0.01
out = np.zeros(3)
y = np.zeros(3)
print("가중치 초기값")
print(w)
print("학습률 : ", lrate)
print("※ learning start")
for epoch in range(10000):
    total_square_sum = 0.0
    for p in range(len(learn_data)):
        pattern_square_sum = 0.0
        for i in range(len(learn_data)):
            out[i] = 0.0
            for j in range(len(learn_data[i])):
                out[i] += w[i][j] * learn_data[p][j]
            y[i] = act_sigmoid(out[i])
        # print(y)
        for i in range(len(learn_data)):
            error = target[p][i] - y[i]
            pattern_square_sum += error ** 2
            for j in range(len(learn_data[p])):
                w[i][j] = w[i][j] + lrate * error * y[i] * (1 - y[i]) * learn_data[p][j]
        total_square_sum += pattern_square_sum
    # print(total_square_sum)
    if total_square_sum < 0.1:
        break
print("※ learning end")
print("############################################################################")
print("학습 횟수: ", epoch)
print("5-4 종료후 학습데이터 및 평가")
print("종료후 학습데이터")
print(w)
print("평가 데이터")
eval_data = [
    [0.0, 1.0, 1.0, 1.0, 1.0,  # G
     1.0, 0.0, 0.0, 0.0, 0.0,
     1.0, 0.0, 0.0, 1.0, 1.0,
     1.0, 0.0, 0.0, 0.0, 1.0,
     0.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 0.0,  # B
     1.0, 0.0, 0.0, 0.0, 1.0,
     1.0, 1.0, 1.0, 1.0, 1.0,
     1.0, 0.0, 0.0, 0.0, 1.0,
     1.0, 1.0, 1.0, 1.0, 0.0],
]
for p in range(len(eval_data)):
    if p == 0:
        print("평가데이터 G")
    else:
        print("평가데이터 B")
    for i in range(5):
        res = ""
        for j in range(5):
            res += str(eval_data[p][i * 5 + j]) + " "
        print(res)
    print()

for p in range(len(eval_data)):
    arr = []
    for i in range(3):
        out[i] = 0.0
        for j in range(25):
            out[i] += w[i][j] * eval_data[p][j]
        y[i] = act_sigmoid(out[i])
        arr.append(y[i])
    result = arr.index(max(arr))
    result_node = ""
    if result == 0:
        result_node = "T"
    elif result == 1:
        result_node = "C"
    else:
        result_node = "E"
        pass
    if p == 0:
        print("eval_data G", ":  ", result_node)
    else:
        print("eval_data B", ":  ", result_node)
