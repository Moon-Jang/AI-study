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
print("3-3 정규화")
print("정규화 된 데이터")
for i in range(len(iris_data.normalized_data)):
    print(iris_data.normalized_data[i])
