import csv
import numpy as np
import random

HEADER = []
# DEGIT = 2
IRIS_DATA = {}
K = 3
NUM = 3


def fetch_excel(path, header):
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = list(csv.reader(f))
        for head in reader[0]:
            header.append(head)
        reader.pop(0)
        data = reader[:]
        return data


def get_distance(a, b, arr_len):
    sum_d = 0
    for i in range(arr_len):
        sum_d += (a[i] - b[i]) ** 2
    return sum_d ** 0.5


def init_means(k):
    m = []
    for i in range(k):
        random_index_x = random.randint(0, 149)
        random_index_y = random.randint(0, 149)
        m.append([sepal_length_list[random_index_x], sepal_width_list[random_index_y]])
    return m


def init_p(k):
    arr = []
    for i in range(K):
        arr.append([])
    return arr


def get_normalization(_list):
    max_element = max(_list)
    min_element = min(_list)
    return list(map(lambda el: (el - min_element) / (max_element - min_element), _list))


def find_min_distance(distances):
    dic_distance = {}
    min_distance = min(distances)
    min_index = distances.index(min_distance)
    dic_distance['target_index'] = min_index
    dic_distance['min_distance'] = min_distance
    return dic_distance


DATA = fetch_excel('./iris_data.csv', HEADER)
print(HEADER)
# print(DATA)
print("pattern_number   output_class    target_class")
sepal_length_list = list(map(lambda el: float(el[0]), DATA))
sepal_width_list = list(map(lambda el: float(el[1]), DATA))
petal_length_list = list(map(lambda el: float(el[2]), DATA))

sepal_length_normalization_list = get_normalization(sepal_length_list)
sepal_width_normalization_list = get_normalization(sepal_width_list)
petal_length_normalization_list = get_normalization(petal_length_list)

location = list(zip(sepal_length_normalization_list, sepal_width_normalization_list, petal_length_normalization_list))

print(location)
means = np.array([[0.2, 0.4, 0.1], [0.3, 0.5, 0.5], [0.7, 0.6, 0.4]])

DATA_size = len(DATA)
epoch_size = 1000
for epoch in range(100):
    p = init_p(K)
    for i in range(DATA_size):
        distances = list(map(lambda mean: get_distance(mean, location[i],NUM), means))
        dic_min_distance = find_min_distance(distances)
        p[dic_min_distance['target_index']].append(location[i][:])
        
        if epoch == epoch_size - 1:
            if i < 50:
                print(f'    {i + 1}             {"A"}               {dic_min_distance["target_index"]}')
            elif i < 100:
                print(f'    {i + 1}             {"B"}               {dic_min_distance["target_index"]}')
            elif i < 150:
                print(f'    {i + 1}             {"C"}               {dic_min_distance["target_index"]}')
    
    for i in range(K):
        new_mean = []
        for j in range(NUM):
            new_mean.append(sum(map(lambda el: el[j], p[i])) / len(p[i]))
            pass
        means[i] = new_mean

print("클래스별 갯수: ", len(p[0]), len(p[1]), len(p[2]))
print("means : ", means)
