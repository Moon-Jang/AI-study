import tensorflow as tf
import numpy as np

print("version", tf.__version__)
INDIM = 26
H1DIM = 13
OUTDIM = 3
LEARN_RATE = 0.01 # 5-2 학습률 초기화

x_data = np.array([
    [1.0,1.0, 1.0, 1.0, 1.0, 1.0,  # T
         0.0, 0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0, 0.0],
    [1.0,1.0, 1.0, 1.0, 1.0, 1.0,  # C
         1.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0,1.0, 1.0, 1.0, 1.0, 1.0,  # E
         1.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 1.0, 1.0, 1.0,
         1.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 1.0, 1.0, 1.0]
])
y_data = np.array([[1,0,0],[0,1,0],[0,0,1]])

X = tf.compat.v1.placeholder(tf.float32, shape=[None, INDIM])
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, OUTDIM])

# 5-1 가중치 초기화
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

for epoch in range(10000):
    cost_val, _ = session.run([cost, train], feed_dict={X: x_data, Y: y_data})
    if cost_val < 0.03:
        break
    if epoch % 1000 == 0:
         print(epoch, "Cost :", cost_val)

test_x_data = np.array([
    [1.0,1.0, 1.0, 1.0, 1.0, 1.0,  # T
         0.0, 0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0, 0.0],
    [1.0,0.0, 1.0, 1.0, 1.0, 1.0,  # C
         1.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 1.0, 1.0, 1.0],
    [1.0,1.0, 1.0, 1.0, 1.0, 1.0,  # E
         1.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 1.0, 1.0, 1.0,
         1.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0,0.0, 1.0, 1.0, 1.0, 1.0,  # G
         1.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 0.0, 0.0, 1.0, 1.0,
         1.0, 0.0, 0.0, 0.0, 1.0,
         0.0, 1.0, 1.0, 1.0, 1.0],
    [1.0,1.0, 1.0, 1.0, 1.0, 0.0,  # B
         1.0, 0.0, 0.0, 0.0, 1.0,
         1.0, 1.0, 1.0, 1.0, 1.0,
         1.0, 0.0, 0.0, 0.0, 1.0,
         1.0, 1.0, 1.0, 1.0, 0.0],
])
test_y_data = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0],[0,0,0]])
result = session.run([y2], feed_dict={X: test_x_data, Y: test_y_data})
print("\n5-5 출력 결과 ")

for el in result[0]:
    print(list(map(lambda el: round(el,2),el)))
