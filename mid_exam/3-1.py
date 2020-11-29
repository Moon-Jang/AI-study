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
