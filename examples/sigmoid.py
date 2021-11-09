import pystar
import pystar.nn as pnn
import pystar.types as pt
import torch
from icecream import ic
import numpy as np
import torch.nn as nn

import matplotlib.pyplot as plt
    
X1_RANGE = (-2, -4)
X2_RANGE = (-6, 4)
X3_RANGE = (3, 5)

c = torch.tensor([0, 0, 0], dtype=float)
V = torch.tensor([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
], dtype=float)

A_ub = torch.tensor([
    [1, 0, 0], 
    [0, 1, 0], 
    [0, 0, 1], 
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, -1]
], dtype=float)

b_ub = torch.tensor([
    max(X1_RANGE),
    max(X2_RANGE),
    max(X3_RANGE),
    -min(X1_RANGE),
    -min(X2_RANGE),
    -min(X3_RANGE)
], dtype=float)

H = pt.HPolytope(A_ub, b_ub)
sl0 = [pt.LinearStarSet(c, V, H)]

a0 = pnn.Sigmoid()
def main():
    sl1 = a0(sl0)
    for s in sl1:
        ic(s.c)
        ic(s.V)
        ic(s.H.A_ub)
        ic(s.H.b_ub)
    plot('First Dimension', sl1[0].H, 0, 3, min(X1_RANGE), max(X1_RANGE))
    plot('Second Dimension', sl1[0].H, 1, 4, min(X2_RANGE), max(X2_RANGE))
    plot('Third Dimension', sl1[0].H, 2, 5, min(X3_RANGE), max(X3_RANGE))
    plt.show()

def plot(title, H, dim_x, dim_y, min_x, max_x):
    verts = vertices(H, dim_x, dim_y, num_directions=200)
    verts.append(verts[0])
    fig = plt.figure()
    plt.title(title)
    x = [p[dim_x] for p in verts]
    y = [p[dim_y] for p in verts]
    plt.plot(x, y)
    x0 = np.linspace(min_x, max_x, 1000)
    y0 = 1/(1 + np.exp(-x0))
    plt.plot(x0, y0)

def vertices(H, dim_x, dim_y, num_directions: int=200):
    verts = []
    last = None
    for theta in np.linspace(0, 2*np.pi, num_directions):
        v = torch.zeros(H.shape, dtype=float)
        v[dim_x] = np.cos(theta)
        v[dim_y] = np.sin(theta)
        p = H.maximize(v).detach().numpy()

        if last is None or not np.allclose(p.flat, last.flat):
            last = p
            verts.append(p)

    return verts

if __name__ == '__main__':
    main()
