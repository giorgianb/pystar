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
c = torch.stack([c, c, c])

V = torch.tensor([[[0], [1], [0], [0]],
    [[0], [0], [0], [0]],
    [[0], [0], [0], [0]],
    [[0], [0], [0], [0]]
    ])
V = torch.stack([V, V, V])

A_ub = torch.tensor([[1], [-1]], dtype=torch.float32)
b_ub = torch.tensor([2, 2], dtype=torch.float32)

H = pt.HPolytope(A_ub, b_ub)
s = pt.LinearStarSet(c, V, H)

ic(c.shape)
ic(V.shape)

mp2d = pnn.MaxPool2d(2, 2)
out = mp2d([s])
for o in out:
    print('-' * 10)
    ic(o.c)
