import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random


class MnistModel:
    def __init__(self):
        self.mnist = tf.keras.datasets.mnist
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.mnist.load_data()
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0
        print("#6-1 첫데이터 10개의 목표값 ")
        print(self.y_train[0:10])
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
            optimizer="adam",
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def learn(self):
        print("Learning Start !!!")
        self.model.fit(self.x_train, self.y_train, batch_size=32, epochs=1, verbose=1)
        print("Learning End !!!")
    
    def get_learn_result(self):
        result_loss, result_accuracy = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        print("#6-2 학습 결과 ")
        print("cost: ", result_loss, "  accuracy:", result_accuracy)
    
    def test(self):
        try:
            print("#6-3")
            _idx = int(input("몇번째 데이터를 보시겠습니까? "))
        except:
            print("숫자가 아닙니다. 다시 시도해주세요")
            import sys
            sys.exit()
            
        x_data = self.x_test[_idx]
        y_data = self.y_test[_idx]
        p = self.model.predict(self.x_test)
        if type(_idx) is int:
            print("목표값", y_data)
            print("#6-4 출력노드")
            print(list(map(lambda el: round(el, 2), p[_idx])))
            plt.imshow(x_data.reshape(28, 28), cmap='Greys', interpolation='nearest')
            plt.show()
        

mnist_model = MnistModel()
mnist_model.learn()
mnist_model.get_learn_result()
mnist_model.test()