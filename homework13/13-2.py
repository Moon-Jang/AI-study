import numpy as np
import tensorflow as tf


class Model:
    def __init__(self, x_train, y_train):
        self.x = x_train
        self.y = y_train
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(3, input_dim=3, activation="relu"),
            tf.keras.layers.Dense(3, activation="relu"),
            tf.keras.layers.Dense(2, activation="softmax"),
        ])
        self.model.summary()
        self.model.compile(
            optimizer='adam',  # SGD
            loss='sparse_categorical_crossentropy',  # mean_squeared_error
            metrics=['accuracy']
        )
        self.x_test = []
        self.init_testcase()
        pass
    
    def init_testcase(self):
        for i in range(11):
            for j in range(11):
                self.x_test.append([1, i / 10, j / 10])
    
    def learn(self):
        print("fit:learning")
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='loss', baseline=0.001, patience=10, verbose=1, mode='min'),
        ]
        self.model.fit(self.x, self.y, epochs=50000, verbose=1, callbacks=callbacks)
    
    def result(self):
        print("prediction")
        print(list(map(lambda el: list(map(lambda el: round(el, 2), el)), self.model.predict(self.x))))
    
    def test(self):
        print(list(map(lambda el: list(map(lambda el: round(el, 2), el)), self.model.predict(self.x_test))))



print(tf.__version__)
xor_xy = np.array([[1.0, 0.0, 0.0, 0.0],
                   [1.0, 0.0, 1.0, 1.0],
                   [1.0, 1.0, 0.0, 1.0],
                   [1.0, 1.0, 1.0, 0.0]])
xor_x_train = xor_xy[:, 0:-1]
xor_y_train = xor_xy[:, -1:]

xor_model = Model(xor_x_train, xor_y_train)
xor_model.learn()
xor_model.result()
xor_model.test()
