import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random


class MnistModel:
    def __init__(self,optimizer,loss):
        self.mnist = tf.keras.datasets.mnist
        self.optimizer = optimizer
        self.loss = loss
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.mnist.load_data()
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0
        self.model = tf.keras.models.Sequential([
            # tf.keras.layers.Flatten(input_shape=(1,3)),
            # tf.keras.layers.Dense(3, activation="sigmoid"),
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(50, input_dim=3, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
        # self.model.summary()
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=['accuracy']
        )
    
    def learn(self):
        print("Learning Start !!!")
        self.model.fit(self.x_train, self.y_train, batch_size=32, epochs=1, verbose=1)
        print("Learning End !!!")
    
    def get_learn_result(self):
        result_loss, result_accuracy = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        print("optimizer : ", self.optimizer, "  loss : ", self.loss)
        print("#결과 ")
        print("result_loss: ", result_loss, "  result_accuracy:", result_accuracy)
    
    def test(self):
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print("test_loss: ", test_loss, "  test_accuracy:", test_accuracy)


optimizer = "adam"
loss = "sparse_categorical_crossentropy"
mnist_model = MnistModel(optimizer,loss)
mnist_model.learn()
mnist_model.get_learn_result()

optimizer = "adam"
loss = "mean_squared_error"
mnist_model = MnistModel(optimizer,loss)
mnist_model.learn()
mnist_model.get_learn_result()

optimizer = "SGD"
loss = "sparse_categorical_crossentropy"
mnist_model = MnistModel(optimizer,loss)
mnist_model.learn()
mnist_model.get_learn_result()

optimizer = "SGD"
loss = "mean_squared_error"
mnist_model = MnistModel(optimizer,loss)
mnist_model.learn()
mnist_model.get_learn_result()