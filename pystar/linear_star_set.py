from __future__ import annotations
import torch
from pystar.types
from pystar.h_polytope import h_polytope

class LinearStarSet:
    def __init__(self: LinearStarSet, c: torch.Tensor, V: torch.Tensor, H: HPolytope) -> None:
        self.c = c
        self.V = V
        self.H = H

    @property
    def shape(self):
        return self.c.shape

    def __and__(self: LinearStarSet, H: HPolytope):
        A_ub = np.array(H.A_ub)
        b_ub = np.array(H.b_ub)
        # Convert to domain
        A_ubp = A_ub @ self.V
        b_ubp = b_ub - A_ub @ self.c
        Hp = HPolytope(A_ubp.tolist(), b_ubp.tolist())
        return LinearStarSet(self.c, self.V, self.H & Hp)

    def __iand__(self: LinearStarSet, H: HPolytope):
        A_ub = np.array(H.A_ub)
        b_ub = np.array(H.b_ub)
        # Convert to domain
        A_ubp = A_ub @ self.V
        b_ubp = b_ub - A_ub @ self.c
        Hp = HPolytope(A_ubp.tolist(), b_ubp.tolist())
        self.H &= Hp
        return self

    def maximize(self: LinearStarSet, v: RowVector) -> ColumVector:
        try:
            domain_dir = v @ self.V
            domain_max = self.H.maximize(domain_dir)
            range_max = self.c + self.V @ domain_max
        except ValueError:
            print("[Linear Star Set]")
            ic(self.V.shape)
            ic(self.c.shape)
            ic(v.shape)
            raise

        return range_max


