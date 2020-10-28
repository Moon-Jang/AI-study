from mid_exam.PerceptronLearnData import learn_data

patterns = ["T","C","E"]

for i in range(len(learn_data)):
    print("pattern:"+patterns[i])
    for j in range(5):
        row_str = ""
        for k in range(5):
            row_str += str(learn_data[i][(j*5)+k]) + " "
        print(row_str)
    print()
