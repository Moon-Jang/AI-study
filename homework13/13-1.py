import numpy as np
import tensorflow as tf


class Model:
    def __init__(self, x_train, y_train):
        self.x = x_train
        self.y = y_train
        self.model = tf.keras.models.Sequential([
            # tf.keras.layers.Flatten(input_shape=(1,3)),
            # tf.keras.layers.Dense(3, activation="sigmoid"),
            tf.keras.layers.Dense(3, input_dim=3, activation="sigmoid"),
            tf.keras.layers.Dense(2, activation="sigmoid"),
        ])
        self.model.summary()
        self.model.compile(
            optimizer='adam',  # SGD
            loss='sparse_categorical_crossentropy',  # mean_squeared_error
            metrics=['accuracy']
        )
        pass
    
    def learn(self):
        print("fit:learning")
        self.model.fit(self.x, self.y, epochs=50000, verbose=0)
    
    def result(self):
        print("prediction")
        print(list(map(lambda el: list(map(lambda el: round(el, 6), el)), self.model.predict(self.x))))


print(tf.__version__)
or_xy = np.array([[1.0, 0.0, 0.0, 0.0],
               [1.0, 0.0, 1.0, 1.0],
               [1.0, 1.0, 0.0, 1.0],
               [1.0, 1.0, 1.0, 1.0]])
or_x_train = or_xy[:, 0:-1]
or_y_train = or_xy[:, -1:]

or_model = Model(or_x_train, or_y_train)
or_model.learn()
or_model.result()

and_xy = np.array([[1.0, 0.0, 0.0, 0.0],
               [1.0, 0.0, 1.0, 0.0],
               [1.0, 1.0, 0.0, 0.0],
               [1.0, 1.0, 1.0, 1.0]])
and_x_train = and_xy[:, 0:-1]
and_y_train = and_xy[:, -1:]
and_model = Model(and_x_train,and_y_train)
and_model.learn()
and_model.result()
