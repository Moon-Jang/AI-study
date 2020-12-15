import tensorflow as tf
import numpy as np
import csv
import math
from functools import reduce


class Iris:
    def __init__(self, total_data):
        self.learn_data = []
        self.eval_data = []
        self.normalized_data = []
        self.setosa = []
        self.versicolor = []
        self.virginica = []
        self.rand_data = []
        self.setup_data(total_data)
    
    def setup_data(self, total_data):
        for i in range(3):
            l_data = []
            e_data = []
            for j in range(50):
                if j < 40:
                    temp = total_data[(i * 50) + j][0:4]
                    if i == 0:
                        temp.append(1)
                        temp.append(0)
                        temp.append(0)
                        l_data.append(temp)
                    elif i == 1:
                        temp.append(0)
                        temp.append(1)
                        temp.append(0)
                        l_data.append(temp)
                    elif i == 2:
                        temp.append(0)
                        temp.append(0)
                        temp.append(1)
                        l_data.append(temp)
                        pass
                else:
                    temp = total_data[(i * 50) + j][0:4]
                    if i == 0:
                        temp.append(1)
                        temp.append(0)
                        temp.append(0)
                        e_data.append(temp)
                    elif i == 1:
                        temp.append(0)
                        temp.append(1)
                        temp.append(0)
                        e_data.append(temp)
                    elif i == 2:
                        temp.append(0)
                        temp.append(0)
                        temp.append(1)
                        e_data.append(temp)
                        pass
                    
            self.learn_data += l_data
            self.eval_data += e_data
            pass
        self.normalized_data = list(
            zip(self.get_normalization([element[0] for element in self.learn_data]),
                self.get_normalization([element[1] for element in self.learn_data]),
                self.get_normalization([element[2] for element in self.learn_data]),
                self.get_normalization([element[3] for element in self.learn_data]),
                [element[4] for element in self.learn_data],
                [element[5] for element in self.learn_data],
                [element[6] for element in self.learn_data]))
        self.eval_data = list(
            zip(self.get_normalization([element[0] for element in self.eval_data]),
                self.get_normalization([element[1] for element in self.eval_data]),
                self.get_normalization([element[2] for element in self.eval_data]),
                self.get_normalization([element[3] for element in self.eval_data]),
                [element[4] for element in self.eval_data],
                [element[5] for element in self.eval_data],
                [element[6] for element in self.eval_data]))
    
    def get_normalization(self, _list):
        max_element = max(_list)
        min_element = min(_list)
        return list(map(lambda el: round((el - min_element) / (max_element - min_element), 4), _list))
    
    def shuffle(self):
        used_idx = set()
        shuffle_idx = []
        data_length = 120
        while len(shuffle_idx) < 120:
            random_idx = np.random.randint(0, data_length)
            if (random_idx in used_idx) == False:
                shuffle_idx.append(random_idx)
                used_idx.add(random_idx)
            pass
        for idx in shuffle_idx:
            self.rand_data.append(self.learn_data[idx])
            
def fetch_excel(path, header):
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = list(csv.reader(f))
        for head in reader[0]:
            header.append(head)
        reader.pop(0)
        data = [[float(item) for item in line[0:4]] for line in reader]
        return data


DATA = fetch_excel('./iris_data.csv', [])

INDIM = 4
H1DIM = 4
OUTDIM = 3
LEARN_RATE = 0.01 # 3-2 학습률 초기화

iris_data = Iris(DATA)

print("# 3-1 정규환된 아이리스 데이터")
for el in iris_data.normalized_data:
    print(el[0:4])
print("정규화 학습 데이터 갯수",len(iris_data.normalized_data))

x_data = [el[0:4] for el in iris_data.normalized_data]
y_data = [el[4:] for el in iris_data.normalized_data]

X = tf.compat.v1.placeholder(tf.float32, shape=[None, INDIM])
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, OUTDIM])

# 3-2 가중치 초기화
W1 = tf.Variable(tf.random_uniform([INDIM, H1DIM], 0, 0.2), name="weight")
B1 = tf.Variable(tf.random_uniform([H1DIM], 0, 0.2), name="weight")

W2 = tf.Variable(tf.random_uniform([H1DIM, OUTDIM], 0, 0.2), name="weight")
B2 = tf.Variable(tf.random_uniform([OUTDIM], 0, 0.2), name="weight")

y1 = tf.sigmoid(tf.matmul(X, W1) + B1)
y2 = tf.sigmoid(tf.matmul(y1, W2) + B2)
# Error
# cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
cost = tf.reduce_mean((Y - y2) ** 2)
# Change Weight
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=LEARN_RATE)
train = optimizer.minimize(cost)

# tf.equal(predicted, Y)
session = tf.compat.v1.Session()
session.run(tf.compat.v1.global_variables_initializer())

for epoch in range(100000): # 3-2 초기화
    cost_val, _ = session.run([cost, train], feed_dict={X: x_data, Y: y_data})
    if cost_val < 0.03:
        break
    if epoch % 1000 == 0:
         print(epoch, "Cost :", cost_val)


test_x_data = [el[0:4] for el in iris_data.eval_data]
test_y_data = [el[4:] for el in iris_data.eval_data]
result = session.run([y2], feed_dict={X: test_x_data, Y: test_y_data})
print("\n# 3-3 평가 데이터 분석")
for i in range(len(test_y_data)):
    print("출력노드 : ",list(map(lambda el: round(el,2),result[0][i]))," 목표값 :", test_y_data[i])

print("\n#3-4 최종인식률")

correct_num = 0

def get_max_index(arr):
    max_value = max(arr)
    for i in range(len(arr)):
        if max_value == arr[i]:
            return i

for i in range(len(test_y_data)):
    result_idx = get_max_index(result[0][i])
    target_idx = get_max_index(test_y_data[i])
    if result_idx == target_idx:
        correct_num = correct_num + 1
        
print("인식률 : ",correct_num / len(test_y_data))