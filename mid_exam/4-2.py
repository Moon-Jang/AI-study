import numpy as np
import csv
import math
from functools import reduce

K = 3
NUM = 2


class Iris:
    def __init__(self, total_data):
        self.learn_data = []
        self.eval_data = []
        self.normalized_data = []
        self.setosa = []
        self.versicolor = []
        self.virginica = []
        self.rand_data = []
        self.setup_data(total_data)
    
    def setup_data(self, total_data):
        for i in range(3):
            l_data = []
            e_data = []
            for j in range(50):
                if j < 40:
                    l_data.append(total_data[(i * 50) + j][2:4])
                else:
                    e_data.append(total_data[(i * 50) + j][2:4])
            if i == 0:
                self.setosa = IrisType(l_data)
            elif i == 1:
                self.versicolor = IrisType(l_data)
            elif i == 2:
                self.virginica = IrisType(l_data)
                pass
            self.learn_data += l_data
            self.eval_data += e_data
            pass
        self.normalized_data = list(
            zip(self.get_normalization([element[0] for element in self.learn_data]),
                self.get_normalization([element[1] for element in self.learn_data])))
    
    def get_normalization(self, _list):
        max_element = max(_list)
        min_element = min(_list)
        return list(map(lambda el: round((el - min_element) / (max_element - min_element), 4), _list))
    
    def shuffle(self):
        used_idx = set()
        shuffle_idx = []
        data_length = 120
        while len(shuffle_idx) < 120:
            random_idx = np.random.randint(0, data_length)
            if (random_idx in used_idx) == False:
                shuffle_idx.append(random_idx)
                used_idx.add(random_idx)
            pass
        for idx in shuffle_idx:
            self.rand_data.append(self.learn_data[idx])


class IrisType:
    def __init__(self, data):
        self.petal_width = {}
        self.petal_length = {}
        self.DEGIT = 6
        self.setup_data(data)
    
    def setup_data(self, data):
        self.petal_width["data"] = list(map(lambda el: float(el[0]), data))
        self.petal_length["data"] = list(map(lambda el: float(el[1]), data))
        self.petal_width["average"] = self.get_average(self.petal_width["data"])
        self.petal_length["average"] = self.get_average(self.petal_length["data"])
        self.petal_width["variance"] = self.get_variance(self.petal_width["data"], self.petal_width["average"])
        self.petal_length["variance"] = self.get_variance(self.petal_length["data"], self.petal_length["average"])
    
    def get_average(self, _list):
        return round(reduce(lambda acc, cur: acc + cur, _list, 0) / len(_list), self.DEGIT)
    
    def get_variance(self, _list, average):
        deviation_list = tuple(map(lambda item: round(average - item, self.DEGIT), _list))
        return round(reduce(lambda acc, cur: acc + cur ** 2,
                            deviation_list, 0) / len(_list), self.DEGIT)
    
    def get_standard_devation(self, variance):
        return round(math.sqrt(variance), self, self.DEGIT)


def get_distance(a, b):
    # print(list(zip(a,b)))
    return (((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2)) ** 0.5


def init_means(k, data):
    m = []
    for i in range(k):
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


def fetch_excel(path, header):
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = list(csv.reader(f))
        for head in reader[0]:
            header.append(head)
        reader.pop(0)
        data = [[float(item) for item in line[0:4]] for line in reader]
        return data


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
        print("epoch", epoch,data_size)
        for i in range(data_size):
            distances = list(map(lambda mean: get_distance(mean, data[i]), means))
            dic_min_distance = find_min_distance(distances)
            p[dic_min_distance['target_index']].append(data[i][:])
        before_mean = means
        learning_complete = [False, False, False]
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
        print("대푯값 : ", means)
        print("cost: ",len(p[0]),len(p[1]),len(p[2]))
        ## 만약 결과값이 10번이상 같을 경우 군집화 종료하는 로직
        is_same = False
        is_same = len(list(filter(lambda el: el == False, learning_complete))) == 0
        if is_same:
            complete_cnt = complete_cnt + 1
        if complete_cnt > 10:
            break


DATA = fetch_excel('./iris_data.csv', [])
iris_data = Iris(DATA)
print("4-1 randData")
iris_data.shuffle()
print("randData")
for i in range(len(iris_data.rand_data)):
    print(iris_data.rand_data[i])
print("배열길이", len(iris_data.rand_data))
randData = iris_data.rand_data
print("############################################################################")
print("4-2 cluster randData")
print("randData")
# print(DATA)
print("pattern_number   output_class    target_class")
means = init_means(K, iris_data.rand_data)  # 0 1 2
clustering(means, iris_data.rand_data, len(iris_data.rand_data))
print("############################################################################")
print("4-3")
print("최종 대푯값 : ", means)
print("############################################################################")
print("4-4 평가 데이터")
print("입력값\t\t  소속 클래스\t\t분류 결과\t\t분류 대푯값")
for i in range(len(iris_data.eval_data)):
    distances = list(map(lambda mean: get_distance(mean, iris_data.eval_data[i]), means))
    dic_min_distance = find_min_distance(distances)
    result_class = ""
    if dic_min_distance['target_index'] == 0:
        result_class = "A"
    elif dic_min_distance['target_index'] == 1:
        result_class = "B"
    else:
        result_class = "C"
    if i < 10:
        print(iris_data.eval_data[i], "\t\tsetosa", "\t\t", result_class,
              "\t\t\t",means[dic_min_distance['target_index']])
    elif i < 20:
        print(iris_data.eval_data[i], "\t\tversicolor", "\t\t", result_class,
              "\t\t\t", means[dic_min_distance['target_index']])
    else:
        print(iris_data.eval_data[i], "\t\tvirginica", "\t\t", result_class,
              "\t\t\t", means[dic_min_distance['target_index']])
