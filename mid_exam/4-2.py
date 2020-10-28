import numpy as np

from mid_exam.Iris import Iris
from mid_exam.myutil import fetch_excel

K = 3
NUM = 2


def get_distance(a, b):
    # print(list(zip(a,b)))
    return (((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2)) ** 0.5


def init_means(k, data):
    m = []
    for i in range(k):
        # random_index_x = np.random.randint(0, len(data))
        # random_index_y = np.random.randint(0, len(data))
        arr = []
        for j in range(len(data[i])):
            random_index = np.random.randint(0, len(data))
            arr.append(data[random_index][j])
        m.append(arr)
    return m


def init_p(k):
    arr = []
    for i in range(K):
        arr.append([])
    return arr


def find_min_distance(distances):
    dic_distance = {}
    min_distance = min(distances)
    min_index = distances.index(min_distance)
    dic_distance['target_index'] = min_index  # 0
    dic_distance['min_distance'] = min_distance  #
    return dic_distance


def clustering(means, data, data_size):
    complete_cnt = 0
    for epoch in range(1000000000):
        p = init_p(K)
        print("epoch",epoch)
        for i in range(data_size):
            distances = list(map(lambda mean: get_distance(mean, data[i]), means))
            dic_min_distance = find_min_distance(distances)
            p[dic_min_distance['target_index']].append(data[i][:])
            # if epoch == 9:
            #     if i < 50:
            #         print(f'    {i + 1}             {"A"}               {dic_min_distance["target_index"]}')
            #     elif i < 100:
            #         print(f'    {i + 1}             {"B"}               {dic_min_distance["target_index"]}')
            #     elif i < 150:
            #         print(f'    {i + 1}             {"C"}               {dic_min_distance["target_index"]}')
            # pass
        before_mean = means
        learning_complete = [False,False,False]
        for i in range(K):
            new_mean = []
            for j in range(NUM):
                if len(p[i]) == 0:
                    random_index = np.random.randint(0, len(data))
                    new_mean.append(data[random_index][j])
                    continue
                new_value = round(sum(map(lambda el: el[j], p[i])) / len(p[i]), 4)
                new_mean.append(new_value)
                pass
            for j in range(NUM):
                learning_complete[i] = before_mean[i][j] == new_mean[j]
                pass
            means[i] = new_mean

        is_same = False
        is_same = len(list(filter(lambda el: el == False, learning_complete))) == 0
        if is_same:
            complete_cnt = complete_cnt + 1
        print("means : ", means)
        if complete_cnt > 10:
            break


DATA = fetch_excel('./iris_data.csv', [])
iris_data = Iris(DATA)
print("4-2 cluster randData")
iris_data.shuffle()
print("randData")
# print(DATA)
print("pattern_number   output_class    target_class")
means = init_means(K, iris_data.rand_data)  # 0 1 2
clustering(means,iris_data.rand_data,len(iris_data.rand_data))
print(means)