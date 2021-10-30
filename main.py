import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import pystar
from icecream import ic

NPOINTS = 5000
X1_RANGE = (0.5, 1)
X2_RANGE = (0, 2)
OVERAPPROX = False
W0 = np.array([
    [np.cos(np.pi/4), -np.sin(np.pi/4)],
    [np.sin(np.pi/4), np.cos(np.pi/4)]
])

b0 = np.array([
    [0],
    [0]
])

W1 = np.identity(2)
b1 = np.array([[0], [-np.sqrt(2)/2]])
INDEX_TO_COLOR = [
        'xkcd:purple',
        'xkcd:green',
        'xkcd:blue',
        'xkcd:pink',
        'xkcd:brown',
        'xkcd:red',
        'xkcd:light blue'
        'xkcd:teal',
        'xkcd:orange',
        'xkcd:magenta',
        'xkcd:yellow',
        'xkcd:sky blue',
        'xkcd:lavender',
        'xkcd:turquoise',
        'xkcd:periwinkle'
        'xkcd:aqua'
]

class Layer:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def __call__(self, v):
        z = self.W @ v + self.b
        a = np.zeros(z.shape[0], dtype=str)
        a[(z < 0).flat] = '-'
        a[(z >= 0).flat] = '+'
        z[z < 0] = 0
        return z, ''.join(a)

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, v):
        aps = []
        for layer in self.layers:
            vp, ap = layer(v)
            aps.append(ap)
            v = vp

        return v, ''.join(aps)

def color(color_index, ap):
    global INDEX_TO_COLOR
    if ap in color_index:
        return INDEX_TO_COLOR[color_index[ap]]
    else:
        index = len(color_index)
        color_index[ap] = index
        return INDEX_TO_COLOR[index]


m = Sequential([
    Layer(W0, b0),
    Layer(W1, b1)
])

mls = pystar.Sequential([
#    pystar.Plot2D('Initial State', 5, domain_x_range=(-2.5, 2.5), domain_y_range=(-2.5, 2.5), range_x_range=(-2.5, 2.5), range_y_range=(-2.5, 2.5)),
    pystar.Dense(W0, b0),
#    pystar.Plot2D('First Dense', 20, domain_x_range=(-2.5, 2.5), domain_y_range=(-2.5, 2.5), range_x_range=(-2.5, 2.5), range_y_range=(-2.5, 2.5)),
    pystar.ReLU(overapprox=OVERAPPROX),
    pystar.Plot2D('First ReLU', 20, domain_x_range=(-2.5, 2.5), domain_y_range=(-2.5, 2.5), range_x_range=(-2.5, 2.5), range_y_range=(-2.5, 2.5)),
    pystar.Dense(W1, b1),
#    pystar.Plot2D('Second Dense', 20, domain_x_range=(-2.5, 2.5), domain_y_range=(-2.5, 2.5), range_x_range=(-2.5, 2.5), range_y_range=(-2.5, 2.5)),
    pystar.ReLU(overapprox=OVERAPPROX),
    pystar.Plot2D('Second ReLU', 20, domain_x_range=(-2.5, 2.5), domain_y_range=(-2.5, 2.5), range_x_range=(-2.5, 2.5), range_y_range=(-2.5, 2.5)),
])

def random_probe():
    color_index = {}
    colors = []
    xs = []
    ys = []
    print(m(np.array([[0.75], [1.0]])))
    for i in range(NPOINTS):
        x1 = np.random.uniform(*X1_RANGE)
        x2 = np.random.uniform(*X2_RANGE)
        x = np.array([[x1], [x2]])
        y, ap = m(x)
        xs.append(y[0])
        ys.append(y[1])
        colors.append(color(color_index, ap))

    plt.title('Random Exploration')
    plt.xlim([-2.5, 2.5])
    plt.ylim([-2.5, 2.5])
    plt.scatter(xs, ys, c=colors)


def exact_analysis():
    c = pystar.column([0, 0])
    V = np.array([
        [1, 0],
        [0, 1]
    ])
    A_ub = [[1, 0], 
            [0, 1], 
            [-1, 0],
            [0, -1]]
    b_ub = [[max(X1_RANGE)], 
            [max(X2_RANGE)], 
            [-min(X1_RANGE)], 
            [-min(X2_RANGE)]]


    H = pystar.HPolytope(A_ub, b_ub)
    sl = [pystar.LinearStarSet(c, V, H)]
    mls(sl)

if __name__ == '__main__':
    #random_probe()
    exact_analysis()
    plt.show()
