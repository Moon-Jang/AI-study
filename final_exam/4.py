import numpy as np
import tensorflow as tf


class Model:
    def __init__(self, x_train, y_train):
        self.x = x_train
        self.y = y_train
        self.model = tf.keras.models.Sequential([
            # tf.keras.layers.Flatten(input_shape=(1,3)),
            # tf.keras.layers.Dense(3, activation="sigmoid"),
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
        print("Learning start !!!")
        self.model.fit(self.x, self.y, epochs=30000, verbose=0)
        print("Learnging End !!!")
    
    def get_learn_result(self):
        result_loss, result_accuracy = self.model.evaluate(self.x, self.y, verbose=0)
        print("#4-1 학습 결과 ")
        print("cost: ", result_loss, "  accuracy:", result_accuracy)
        
    def result(self):
        print("prediction")
        print(list(map(lambda el: list(map(lambda el: round(el, 2), el)), self.model.predict(self.x))))
    
    def test(self):
        print("#4-2")
        out = list(map(lambda el: list(map(lambda el: round(el, 2), el)), self.model.predict(self.x_test)))
        for i in range(11):
            for j in range(11):
                print("[", i / 10, ",", j / 10, "] =>", out[(i*11)+j])


print(tf.__version__)
xor_xy = np.array([[1.0, 0.0, 0.0, 0.0],
                   [1.0, 0.0, 1.0, 1.0],
                   [1.0, 1.0, 0.0, 1.0],
                   [1.0, 1.0, 1.0, 0.0]])
xor_x_train = xor_xy[:, 0:-1]
xor_y_train = xor_xy[:, -1:]

xor_model = Model(xor_x_train, xor_y_train)
xor_model.learn()
xor_model.get_learn_result()
xor_model.result()
xor_model.test()
