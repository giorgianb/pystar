from __future__ import annotations
from pystar.types import ListMatrix, ListVector
from scipy.optimize import linprog

class HPolytope:
    def __init__(self: HPolytope, A_ub: ListMatrix, b_ub: ListVector) -> None:
        self.A_ub = A_ub
        self.b_ub = b_ub

    def __and__(self: HPolytope, other: HPolytope) -> HPolytope:
        """Perform intersection with another H-Polytope"""
        A_ub = self.A_ub + other.A_ub
        b_ub = self.b_ub + other.b_ub

        return HPolytope(A_ub, b_ub)

    def __iand__(self: HPolytope, H: HPolytope) -> None:
        """Perform in-place intersection with another H-Polytope"""
        self.A_ub.extend(deepcopy(H.A_ub))
        self.b_ub.extend(b_ub)
        return self

    def __matmul__(self: HPolytope, v: RowVector) -> torch.Tensor:
        """Maximize along the direction specified by 'v'."""
        try:
            res = linprog(-1 * v, A_ub=self.A_ub, b_ub=self.b_ub, bounds=(-np.inf, np.inf))
            assert res.success
        except AssertionError:
            print("[HPolytope]")
            ic(v)
            ic(self.A_ub)
            ic(self.b_ub)
            raise

        return res
