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
print("정규화 된 데이터")
for i in range(len(iris_data.normalized_data)):
    print(iris_data.normalized_data[i])

