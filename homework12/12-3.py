import tensorflow as tf
import numpy as np

print("version", tf.__version__)
INDIM = 26
OUTDIM = 2
LEARN_RATE = 0.03
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
         1.0, 1.0, 1.0, 1.0, 1.0]
])
y_data = np.array([[1,0],[0,1]])

X = tf.compat.v1.placeholder(tf.float32, shape=[None, INDIM])
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, OUTDIM])
W = tf.Variable(tf.random.normal([INDIM, OUTDIM]), name="weight")
B = tf.Variable(tf.random.normal([OUTDIM]), name="weight")

hypothesis = tf.sigmoid(tf.matmul(X, W) + B)
# Error
# cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
cost = tf.reduce_mean((Y - hypothesis) ** 2)
# Change Weight
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=LEARN_RATE)
train = optimizer.minimize(cost)

# tf.equal(predicted, Y)
session = tf.compat.v1.Session()
session.run(tf.compat.v1.global_variables_initializer())

for epoch in range(100000):
    cost_val, _ = session.run([cost, train], feed_dict={X: x_data, Y: y_data})
    if cost_val < 0.01:
        break
    if epoch % 1000 == 0:
         print(epoch, "Cost :", cost_val)

result = session.run([hypothesis], feed_dict={X: x_data, Y: y_data})
print("\n12-3 RESULT \n", result)
