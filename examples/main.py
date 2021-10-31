import pystar
import pystar.nn as pnn
import pystar.types as pt
import torch
from icecream import ic
import numpy as np
import torch.nn as nn

NPOINTS = 5000
X1_RANGE = (0.5, 1)
X2_RANGE = (0, 2)
OVERAPPROX = False
W0 = torch.tensor([
    [np.cos(np.pi/4), -np.sin(np.pi/4)],
    [np.sin(np.pi/4), np.cos(np.pi/4)]
])

b0 = torch.tensor([0, 0], dtype=float)

W1 = torch.tensor([[1, 0], [0, 1]], dtype=float)
b1 = torch.tensor([0, -np.sqrt(2)/2], dtype=float)

d0 = pnn.Linear(2, 2, dtype=float)
r0 = pnn.ReLU(overapprox=OVERAPPROX)
d1 = pnn.Linear(2, 2, dtype=float)
r1 = pnn.ReLU(overapprox=OVERAPPROX)

with torch.no_grad():
    d0.W[:] = W0
    d0.b[:] = b0
    d1.W[:] = W1
    d1.b[:] = b1

c = torch.tensor([0, 0], dtype=float)
V = torch.tensor([
    [1, 0],
    [0, 1]
], dtype=float)
A_ub = torch.tensor([
    [1, 0], 
    [0, 1], 
    [-1, 0],
    [0, -1]
], dtype=float)

b_ub = torch.tensor([
    max(X1_RANGE),
    max(X2_RANGE),
    -min(X1_RANGE),
    -min(X2_RANGE)
], dtype=float)

H = pt.HPolytope(A_ub, b_ub)
sl0 = [pt.LinearStarSet(c, V, H)]

print("[First Linear Layer]")
sl1 = d0(sl0)
for s in sl1:
    ic(s.c)
    ic(s.V)
    ic(s.H.A_ub)
    ic(s.H.b_ub)

print("[First ReLU Layer]")
sl2 = r0(sl1)
for s in sl2:
    ic(s.c)
    ic(s.V)
    ic(s.H.A_ub)
    ic(s.H.b_ub)

print("[Second Linear Layer]")
sl3 = d1(sl2)
for s in sl3:
    ic(s.c)
    ic(s.V)
    ic(s.H.A_ub)
    ic(s.H.b_ub)


print("[Second ReLU Layer]")
sl4 = r1(sl3)
for s in sl4:
    ic(s.c)
    ic(s.V)
    ic(s.H.A_ub)
    ic(s.H.b_ub)

m = nn.Sequential(d0, r0, d1, r1)
print("[Full Model]")
sln = m(sl0)

for s in sl4:
    ic(s.c)
    ic(s.V)
    ic(s.H.A_ub)
    ic(s.H.b_ub)

