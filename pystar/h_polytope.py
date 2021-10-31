from __future__ import annotations
from scipy.optimize import linprog

class HPolytope:
    def __init__(self: HPolytope, A_ub: torch.Tensor, b_ub: torch.Tensor) -> None:
        self.A_ub = A_ub
        self.b_ub = b_ub

    def __and__(self: HPolytope, other: HPolytope) -> HPolytope:
        """Perform intersection with another H-Polytope"""
        A_ub = torch.cat((self.A_ub, other.A_ub), dim=0)
        b_ub = torch.cat((self.b_ub, other.b_ub), dim=0)

        return HPolytope(A_ub, b_ub)

    def __iand__(self: HPolytope, H: HPolytope) -> None:
        """Perform in-place intersection with another H-Polytope"""
        self.A_ub = torch.cat((self.A_ub, other.A_ub), dim=0)
        self.b_ub = torch.cat((self.b_ub, other.b_ub), dim=0)
        return self

    def __matmul__(self: HPolytope, v: torch.Tensor) -> torch.Tensor:
        """Maximize along the direction specified by 'v'."""
        try:
            res = linprog(-1 * v, A_ub=self.A_ub, b_ub=self.b_ub, bounds=(-torch.inf, torch.inf))
            assert res.success
        except AssertionError:
            print("[HPolytope]")
            ic(v)
            ic(self.A_ub)
            ic(self.b_ub)
            raise

        return res.x

    def clone(self: HPolytope) -> HPolytope:
        """
        Performs a shallow clone of the H-Polytope

        No instance methods modify A_ub or b_ub without allocating
        new instances, so a shallow clone is enough in most cases.
        """
        return HPolytope(self.A_ub, self.b_ub)
