import numpy as np
from itertools import groupby

a = np.array([
    [0, 1, 2, 2, 1, 1],
    [0, 0, 0, 0, 1, 1],
    [0, 2, 2, 2, 2, 2],
    [0, 1, 1, 0, 1, 1],
    [0, 1, 1, 0, 1, 1],
    [0, 1, 1, 0, 1, 1]])

for i in range(6):
    dic = {}
    for k, g in groupby(a[i, :]):
        dic[k] = sum(1 for n in g)
    if 1 in dic and dic[1] >= 5:
        print("black")
    elif 2 in dic and dic[2] >= 5:
        print("white")
    dic = {}
    for k, g in groupby(a[:, i]):
        dic[k] = sum(1 for n in g)
    if 1 in dic and dic[1] >= 5:
        print("black")
    elif 2 in dic and dic[2] >= 5:
        print("white")