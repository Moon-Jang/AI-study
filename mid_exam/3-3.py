import csv
import math
from functools import reduce
from mid_exam.Iris import Iris
from mid_exam.myutil import fetch_excel

DATA = fetch_excel('./iris_data.csv', [])
iris_data = Iris(DATA)
print("3-1 learnData / evalData")
print("learnData")
for i in range(len(iris_data.learn_data)):
    print(iris_data.learn_data[i])
print("배열길이", len(iris_data.learn_data))
print("\n=============================================\n")
print("evalData")
for i in range(len(iris_data.eval_data)):
    print(iris_data.eval_data[i])
print("배열길이", len(iris_data.eval_data))
print("############################################################################")
print("3-2 3가지 꽃의 평균과 분산")
print("평균           setosa       ", "versicolor       ", "virginica")
print("petal.width   ", iris_data.setosa.petal_width["average"],
      "        ", iris_data.versicolor.petal_width["average"],
      "          ", iris_data.virginica.petal_width["average"])
print("petal.length  ", iris_data.setosa.petal_length["average"],
      "        ", iris_data.versicolor.petal_length["average"],
      "           ", iris_data.virginica.petal_length["average"])
print("분산           setosa       ", "versicolor     ", "virginica")
print("petal.width   ", iris_data.setosa.petal_width["variance"],
      "      ", iris_data.versicolor.petal_width["variance"],
      "        ", iris_data.virginica.petal_width["variance"])
print("petal.length  ", iris_data.setosa.petal_length["variance"],
      "      ", iris_data.versicolor.petal_length["variance"],
      "         ", iris_data.virginica.petal_length["variance"])
print("############################################################################")
print("3-3 정규화")
print("정규화 된 데이터 길이",len(iris_data.normalized_data))
for i in range(len(iris_data.normalized_data)):
    print(iris_data.normalized_data[i])

