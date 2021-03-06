from __future__ import annotations
import torch
from pystar.types import LinearStarSetBatch
from pystar.types import LinearStarSet, HPolytope
from itertools import product

class ReLU(torch.nn.Module):
    def __init__(self: ReLU, inplace: bool=False, overapprox: bool=False, overapprox_dims: int=1):
        super().__init__()
        self.inplace = inplace
        self.overapprox = overapprox
        self.overapprox_dims = overapprox_dims
        if self.overapprox:
            self.process = self._overapprox
        else:
            self.process = self._exact

    def _overapprox(self: ReLU, s: LinearStarSet):
        for dim in product(*map(range, s.shape)):
            v = torch.zeros(s.shape, dtype=s.dtype)
            v[dim] = -1
            res = s.maximize(v)
            l = res[dim]
            if l >= 0:
                continue
            res = s.maximize(-v)
            u = res[dim]

            Vi = s.V[dim] # Get the generator for the dim
            ci = s.c[dim] # Get the value for dim

            if self.inplace:
                s.V[dim] = 0
                s.c[dim] = 0
            else:
                V = s.V.clone()
                c = s.c.clone()
                V[dim] = 0
                c[dim] = 0
                s = LinearStarSet(c, V, s.H.clone())

            if u > 0:
                d = u/(u - l)
                ei = torch.zeros(s.shape, dtype=s.dtype)
                ei[dim] = 1

                # in-place handling was already done
                # so we already have a new instance if we
                # don't want to do it in-place
                s.V = torch.cat((s.V, ei.unsqueeze(-1)), -1)
                pad = torch.zeros(s.H.A_ub.shape[0], 1)
                s.H.A_ub = torch.cat((s.H.A_ub, pad), axis=1)
                A_ub = torch.zeros(3, s.H.A_ub.shape[1], dtype=s.H.A_ub.dtype)
                A_ub[0, -1] = -1  # First contraints: alpha_i >= 0
                A_ub[1, :-1] = Vi
                A_ub[1, -1] = -1
                A_ub[2, :-1] = -d*Vi
                A_ub[2, -1] = 1
                b_ub = torch.tensor([0, -ci, d*(ci - l)], dtype=s.H.b_ub.dtype)
                H = HPolytope(A_ub, b_ub)
                s.H &= H
        return [s]

    def _tensor_forward(self, input: torch.Tensor):
        if self.inplace:
            input[input < 0] = 0
        else:
            input = input.clone()
            input[input < 0] = 0

        return input

    def _linear_star_set_forward(self, input: LinearStarSetBatch):
        res = []
        for s in input:
            ss = self.process(s)
            res.extend(ss)
        return res

    def _exact(self: ReLU, s: LinearStarSet):
        slp = [s]
        for dim in product(*map(range, s.shape)):
            slp = self._step(slp, dim)

        return slp

    def _step(self: LinearStarSet, sl: LinearStarSetBatch, dim: tuple[int]):
        slp = []
        for s in sl:
            v = torch.zeros(s.shape, dtype=s.dtype)
            v[dim] = -1
            res = s.maximize(v)
            if res[dim] >= 0:
                if self.inplace:
                    slp.append(s)
                else:
                    slp.append(LinearStarSet(s.c, s.V, s.H.clone()))
            else:
                # First case: x_dim == 0
                A_ub = torch.zeros((1, *s.shape), dtype=s.H.A_ub.dtype)
                b_ub = torch.tensor([0], dtype=s.H.b_ub.dtype)
                A_ub[0, dim] = 1
                A_ub = torch.flatten(A_ub, start_dim=1, end_dim=-1)
                s1 = s & HPolytope(A_ub, b_ub) # Restrict domain to x_i <= 0

                Vp = s.V.clone()
                cp = s.c.clone()
                Vp[dim] = 0
                cp[dim] = 0
                s1 = LinearStarSet(cp, Vp, s1.H)
                slp.append(s1)

                # Second case: x_dim > 0
                res = s.maximize(-v)
                if res[dim] > 0:
                    A_ub = torch.zeros((1, *s.shape), dtype=s.H.A_ub.dtype)
                    b_ub = torch.tensor([0], dtype=s.H.b_ub.dtype)
                    A_ub[0, dim] = -1
                    if self.inplace:
                        s &= HPolytope(A_ub, b_ub)
                        s2 = s
                    else:
                        s2 = s & HPolytope(A_ub, b_ub)
                    slp.append(s2)
        return slp

    def forward(self, input: Union[LinearStarSetBatch, torch.Tensor]):
        if type(input) == torch.Tensor:
            return self._tensor_forward(input)
        elif type(input) == list:
            return self._linear_star_set_forward(input)
        else:
            raise TypeError("Input must be either torch.Tensor or pystar.types.LinearStarSetBatch")
