import numpy as np
import tensorflow as tf


class Model:
    def __init__(self, x_train, y_train):
        self.x = x_train
        self.y = y_train
        self.model = tf.keras.models.Sequential([
            # tf.keras.layers.Flatten(input_shape=(1,3)),
            # tf.keras.layers.Dense(3, activation="sigmoid"),
            tf.keras.layers.Dense(26, input_dim=26, activation="relu"),
            tf.keras.layers.Dense(17, activation="relu"),
            tf.keras.layers.Dense(9, activation="relu"),
            tf.keras.layers.Dense(2, activation="softmax"),
        ])
        self.model.summary()
        self.model.compile(
            optimizer='adam',  # SGD
            loss='categorical_crossentropy',  # mean_squeared_error
            metrics=['accuracy']
        )
        pass
    
    def learn(self):
        print("fit:learning")
        self.model.fit(self.x, self.y, epochs=1000, verbose=1)
    
    def result(self):
        print("prediction")
        print(list(map(lambda el: list(map(lambda el: round(el, 2), el)), self.model.predict(self.x))))


print(tf.__version__)
x_train = np.array([
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # T
     0.0, 0.0, 1.0, 0.0, 0.0,
     0.0, 0.0, 1.0, 0.0, 0.0,
     0.0, 0.0, 1.0, 0.0, 0.0,
     0.0, 0.0, 1.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # C
     1.0, 0.0, 0.0, 0.0, 0.0,
     1.0, 0.0, 0.0, 0.0, 0.0,
     1.0, 0.0, 0.0, 0.0, 0.0,
     1.0, 1.0, 1.0, 1.0, 1.0]
])
y_train = np.array([[1, 0], [0, 1]])

model = Model(x_train, y_train)
model.learn()
model.result()
