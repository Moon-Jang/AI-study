import csv
import string
from functools import reduce
import math

HEADER = []
DATA = []
DEGIT = 2
IRIS_DATA = {}


def fetchExcelData(path, header, data):
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = list(csv.reader(f))
        for head in reader[0]:
            header.append(head)
        reader.pop(0)
        for data in reader:
            DATA.append(data)


def getAverage(_list):
    return round(reduce(lambda acc, cur: acc + cur, _list, 0) / len(_list), DEGIT)


def getVariance(_list):
    average = getAverage(_list)
    deviationList = tuple(map(lambda item: round(average - item, DEGIT), _list))
    return round(reduce(lambda acc, cur: acc + cur ** 2,
                        deviationList, 0) / len(_list), DEGIT)


def getStandardDevation(_list):
    variance = getVariance(_list)
    return round(math.sqrt(variance), DEGIT)


fetchExcelData('./iris_data.csv', HEADER, DATA)

for i in range(len(HEADER) - 1):
    IRIS_DATA[HEADER[i]] = {
        'total': list(map(lambda el: float(el[i]), DATA)),
        'Setosa': list(map(lambda el: float(el[i]),
                           list(filter(lambda el: el[4] == 'Setosa', DATA)))
                       ),
        'Versicolor': list(map(lambda el: float(el[i]),
                               list(filter(lambda el: el[4] == 'Versicolor', DATA)))
                           ),
        'Virginica': list(map(lambda el: float(el[i]),
                              list(filter(lambda el: el[4] == 'Virginica', DATA)))
                          )
    }

statistics = {}
keys = list(IRIS_DATA['sepal.length'].keys())

for i in range(len(HEADER) - 1):
    statistics[HEADER[i]] = {}
    for key in keys:
        statistics[HEADER[i]][key] = {
            'max': max(IRIS_DATA[HEADER[i]][key]),
            'min': min(IRIS_DATA[HEADER[i]][key])
        }

for key1 in statistics.keys():
    print(key1)
    for key2 in statistics[key1].keys():
        print(f'    - {key2}')
        cnt = 1
        for key3 in statistics[key1][key2].keys():
            print(f'        {cnt}) {key3}: {statistics[key1][key2][key3]}')
            cnt += 1
