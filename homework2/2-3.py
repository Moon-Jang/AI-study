from functools import reduce
import math

DEGIT = 2


def getAverage(_list):
    return reduce(lambda acc, cur: acc + cur, _list, 0) / len(_list)


def getVariance(_list):
    deviationList = tuple(map(lambda item: round(AVG-item, DEGIT), _list))
    return round(reduce(lambda acc, cur: acc + cur**2,
                        deviationList, 0) / len(_list), DEGIT)


def getStandardDevation(variance):
    return round(math.sqrt(variance), DEGIT)


heightList = (173, 188, 160, 157, 178, 158, 196, 177, 166, 176)

AVG = getAverage(heightList)
print("키 평균 :", AVG)

VARIANCE = getVariance(heightList)
print("키 분산 :", VARIANCE)

STANDARD_DEVATION = getStandardDevation(VARIANCE)
print("키 표준편차 :", STANDARD_DEVATION)

weightList = (77, 85, 57, 47, 72, 55, 87, 88, 67, 81)

AVG = getAverage(weightList)
print("몸무게 평균 :", AVG)

VARIANCE = getVariance(weightList)
print("몸무게 분산 :", VARIANCE)

STANDARD_DEVATION = getStandardDevation(VARIANCE)
print("몸무게 표준편차 :", STANDARD_DEVATION)
