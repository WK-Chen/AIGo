import numpy as np
from collections import Counter

data = np.array([[0,0,2],
                 [0,2,2],
                 [1,1,1]])

print(data.reshape(-1))

print(data[1, 1])