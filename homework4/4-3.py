import csv
import numpy as np
import random

HEADER = []
# DEGIT = 2
IRIS_DATA = {}
K = 3
NUM = 4


def fetch_excel(path, header):
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = list(csv.reader(f))
        for head in reader[0]:
            header.append(head)
        reader.pop(0)
        data = reader[:]
        return data


def get_distance(a, b):
    # print(list(zip(a,b)))
    return (((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2) + ((a[2] - b[2]) ** 2) + ((a[3] - b[3]) ** 2)) ** 0.5


# def init_means(k):
#     m = []
#     for i in range(k):
#         random_index_x = random.randint(0, 149)
#         random_index_y = random.randint(0, 149)
#         m.append([sepal_length_list[random_index_x], sepal_width_list[random_index_y]])
#     return m


def init_p(k):
    arr = []
    for i in range(K):
        arr.append([])
    return arr


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
petal_width_list = list(map(lambda el: float(el[3]), DATA))
location = list(zip(sepal_length_list, sepal_width_list, petal_length_list, petal_width_list))

means = np.array([[5.2, 4.0, 1.4, 0.2], [5.0, 3.3, 3.5, 1.3], [4.7, 3.5, 5.5, 2.1]])
p = init_p(K)

DATA_size = len(DATA)
epoch_size = 1000
for epoch in range(epoch_size):
    p = init_p(K)
    for i in range(DATA_size):
        distances = list(map(lambda mean: get_distance(mean, location[i]), means))
        # print(distances)
        dic_min_distance = find_min_distance(distances)
        # print(dic_min_distance)
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
