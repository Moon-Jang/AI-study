import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print(x_train.shape)
print(y_train.shape)
print("first target : ", y_train[0])

model = tf.keras.models.Sequential([
    # tf.keras.layers.Flatten(input_shape=(1,3)),
    # tf.keras.layers.Dense(3, activation="sigmoid"),
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(50, input_dim=3, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.summary()

model.compile(
    optimizer="adam",
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Learning Start !!!")

model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)

print("Evaluation Start !!!")

result_loss, result_accuracy = model.evaluate(x_train, y_train, verbose=0)
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("result_loss: ",result_loss,"  result_accuracy:",result_accuracy)
print(model.evaluate(x_train, y_train, verbose=0))
print("test loss: ",loss,"  test accuracy:",accuracy)

print("Prediciton Results !!!")
p = model.predict(x_test)

print("y_test[0]",y_test[0])
print("p[0]",p[0])
print("y_test[1]",y_test[1])
print("y_test[9999]",y_test[9999])
print("p[9999]",p[9999])