import numpy as np
from icecream import ic
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def dsigmoid(x):
    return np.exp(-x)/(1 + np.exp(-x))**2


def find_point_of_average_derivative(x1, x2):
    y1 = sigmoid(x1)
    y2 = sigmoid(x2)

    slope = (y2 - y1)/(x2 - x1)
    options = np.roots([slope, 2*slope - 1, slope]).real
    ret = []
    for option in options:
        if option < 0:
            continue

        x = -np.log(option)
        if x1 <= x <= x2:
            ret.append(x)

    return ret
