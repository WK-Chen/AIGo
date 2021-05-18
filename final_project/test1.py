import numpy as np
x = [(None, None), ((np.rot90, 1), None), ((np.rot90, 2), None),
                ((np.rot90, 3), None), (np.fliplr, None), (np.flipud, None),
                (np.flipud, (np.rot90, 1)), (np.fliplr, (np.rot90, 1))]

o = np.array([[1,2,3],[4,5,6],[7,8,9]])
y = np.reshape(o, -1)
print(o)
print(y)
b = np.rot90(o)
print(b)