import csv
import numpy as np
import random


def fetch_excel(path, header):
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = list(csv.reader(f))
        for head in reader[0]:
            header.append(head)
        reader.pop(0)
        data = reader[:]
        return data


# nx = (x - 23.7) / (57.2 - 23.7)
def get_normalization(_list):
    max_element = max(_list)
    min_element = min(_list)
    return list(map(lambda el: (el - min_element) / (max_element - min_element), _list))


DATA = fetch_excel('./iris_data.csv', [])
sepal_length_list = list(map(lambda el: float(el[0]), DATA))
sepal_width_list = list(map(lambda el: float(el[1]), DATA))
petal_length_list = list(map(lambda el: float(el[2]), DATA))
petal_width_list = list(map(lambda el: float(el[3]), DATA))

sepal_length_normalization_list = get_normalization(sepal_length_list)
sepal_width_normalization_list = get_normalization(sepal_width_list)
petal_length_normalization_list = get_normalization(petal_length_list)
petal_width_normalization_list = get_normalization(petal_width_list)

print("sepal_length_normalization_list", sepal_length_normalization_list)
print("sepal_width_normalization_list", sepal_width_normalization_list)
print("petal_length_normalization_list", petal_length_normalization_list)
print("petal_width_normalization_list", petal_width_normalization_list)

