from mid_exam.Iris import Iris
from mid_exam.myutil import fetch_excel

HEADER = []
K = 3
NUM = 2

DATA = fetch_excel('./iris_data.csv', HEADER)
iris_data = Iris(DATA)
print("4-1 randData")
iris_data.shuffle()
print("randData")
for i in range(len(iris_data.rand_data)):
    print(iris_data.rand_data[i])
print("배열길이", len(iris_data.rand_data))
randData = iris_data.rand_data
