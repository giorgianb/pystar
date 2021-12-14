import pystar.nn as pnn
from icecream import ic
import pystar.types as pt
import torch

c = torch.tensor([
        [0, 4, 1, 2],
        [2, 3, 2, 3],
        [1, 3, 1, 2],
        [2, 1, 3, 2]
], dtype=torch.float32)

V = torch.tensor([[[0], [1], [0], [0]],
    [[0], [0], [0], [0]],
    [[0], [0], [0], [0]],
    [[0], [0], [0], [0]]
    ])

A_ub = torch.tensor([[1], [-1]], dtype=torch.float32)
b_ub = torch.tensor([2, 2], dtype=torch.float32)

H = pt.HPolytope(A_ub, b_ub)
s = pt.LinearStarSet(c, V, H)

ic(c.shape)
ic(V.shape)

mp2d = pnn.MaxPool2d(2, 2, overapprox=True)
out = mp2d([s])[0]
ic(out.c)
ic(out.V)
ic(out.H.A_ub)
ic(out.H.b_ub)

