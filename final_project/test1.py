import torch
import re
import numpy as np


def softmax_ignore_zero(x):
    flag = np.array(x > 0)
    exp_x = np.exp(x - np.max(x)) * flag
    exp_x /= np.sum(exp_x)
    return exp_x


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


x = torch.tensor([[-0.1285],
                  [-0.0333],
                  [-0.0588],
                  [0.0403],
                  [-0.0526],
                  [0.0526]])

y = torch.tensor([[-1],
                  [-1],
                  [-1],
                  [1],
                  [-1],
                  [1],])
mse = torch.nn.MSELoss()
z = (x-y)**2
print(z.mean())
print(mse(x.view(-1), y.view(-1)))