import csv
import math
from functools import reduce
from mid_exam.Iris import Iris


def fetch_excel(path, header):
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = list(csv.reader(f))
        for head in reader[0]:
            header.append(head)
        reader.pop(0)
        data = [[float(item) for item in line[0:4]] for line in reader]
        return data


DATA = fetch_excel('./iris_data.csv', [])
iris_data = Iris(DATA)
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
