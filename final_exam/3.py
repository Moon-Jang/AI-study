import numpy as np
import csv
import math
from functools import reduce

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
                    temp = total_data[(i * 50) + j][0:4]
                    if i == 0:
                        temp.append(1)
                        temp.append(0)
                        temp.append(0)
                        l_data.append(temp)
                    elif i == 1:
                        temp.append(0)
                        temp.append(1)
                        temp.append(0)
                        l_data.append(temp)
                    elif i == 2:
                        temp.append(0)
                        temp.append(0)
                        temp.append(1)
                        l_data.append(temp)
                        pass
                else:
                    e_data.append(total_data[(i * 50) + j][0:4])
            
            self.learn_data += l_data
            self.eval_data += e_data
            pass
        self.normalized_data = list(
            zip(self.get_normalization([element[0] for element in self.learn_data]),
                self.get_normalization([element[1] for element in self.learn_data]),
                self.get_normalization([element[2] for element in self.learn_data]),
                self.get_normalization([element[3] for element in self.learn_data]),
                self.get_normalization([element[4] for element in self.learn_data]),
                self.get_normalization([element[5] for element in self.learn_data]),
                self.get_normalization([element[6] for element in self.learn_data])))
        self.eval_data = list(
            zip(self.get_normalization([element[0] for element in self.eval_data]),
                self.get_normalization([element[1] for element in self.eval_data]),
                self.get_normalization([element[2] for element in self.eval_data]),
                self.get_normalization([element[3] for element in self.eval_data]),))
    
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

print("# 3-1 정규환된 아이리스 데이터")
for el in iris_data.normalized_data:
    print(el[0:4])
