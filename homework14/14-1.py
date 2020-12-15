import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/ 255.0, x_test/ 255.0
print(x_train.shape)
print(y_train.shape)
print("first target : ", y_train[0])

str =  input("학습 = Y, 테스트 = N (default Y) \nwhat is this? ")

x_data = -1
y_data = -1

try:
    _idx = int(input("몇번째 데이터를 보시겠습니까? "))
except:
    print("숫자가 아닙니다. 다시 시도해주세요")
    import sys
    sys.exit()
    
if str == 'Y':
    x_data = x_train[_idx]
    y_data = y_train[_idx]
else:
    x_data = x_test[_idx]
    y_data = y_test[_idx]
    
if type(_idx) is int:
    print("test",y_data)
    plt.imshow(x_data.reshape(28,28), cmap='Greys', interpolation='nearest')
    plt.show()