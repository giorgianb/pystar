from __future__ import annotations
import torch
from pystar.h_polytope import HPolytope

class LinearStarSet:
    def __init__(self: LinearStarSet, c: torch.Tensor, V: torch.Tensor, H: HPolytope, dtype=None) -> None:
        """
        Represents a linear star set.

        c is the anchor tensor.
        V is the generator matrix. It has shape c.shape + (n_generators,).
        It has one additional dimension over c, and the last dimension is used to index
        the generator. V[..., i] has the same shape as c and is the i-th generator.
        H is the domain H-polytope. It places constraints on the coefficients of the generators.
        The domain must have the same number of dimensions as the number of generators.

        Here are the two common cases:

        c is a 1-D tensor of shape [d]
        V is a 2-D tensor of shape [d, n_generators]
        H is an H-polytope of space with n_generator dimensions.
        A_ub has dimensions [n_constraints, n_generators] 
        b_ub has dimensions [n_constraints]

        c is a 3-D tensor of shape [H, W, C]
        V is a 2-D tensor of shape [H, W, C, n_generators]
        H is an H-polytope of space with n_generator dimensions.
        A_ub has dimensions [n_constraints, n_generators] 
        b_ub has dimensions [n_constraints]
        """
        if dtype is None:
            dtype = c.dtype
        self.c = c.to(dtype)
        self.V = V.to(dtype)
        self.H = H

    @property
    def dtype(self: LinearStarSet):
        return self.c.dtype

    @property
    def shape(self: LinearStarSet):
        return self.c.shape

    def to(self: LinearStarSet, dtype) -> LinearStarSet:
        cp = self.c.to(dtype)
        Vp = self.c.to(dtype)
        return LinearStarSet(cp, Vp, H.clone())

    @property
    def n_generators(self):
        return self.V.shape[-1]

    def __and__(self: LinearStarSet, H: HPolytope):
        # Convert to domain
        A_ubp = H.A_ub @ self.V
        b_ubp = H.b_ub - H.A_ub @ self.c
        Hp = HPolytope(A_ubp, b_ubp)
        return LinearStarSet(self.c, self.V, self.H & Hp)

    def __iand__(self: LinearStarSet, H: HPolytope):
        # Convert to domain
        # A_ubp: [n_constraints, n_generators]
        # A_ub: [n_constraints, d]
        # V: [d, n_generators]
        A_ubp = H.A_ub @ self.V
        b_ubp = H.b_ub - H.A_ub @ self.c

        Hp = HPolytope(A_ubp, b_ubp)
        self.H &= Hp
        return self

    def maximize(self: LinearStarSet, v: torch.Tensor) -> torch.Tensor:
        """
        Find the maximum extent of the LinearStarSet along a particular direction defined by 'v'.
        'v' has the same shape as c.shape

        Strictly speaking, this maximizes the function v @ x, where x is a coordinate
        within the star set.
        """
        try:
            domain_dir = torch.flatten(v) @ torch.flatten(self.V, end_dim=-2)
            domain_max = self.H.maximize(domain_dir)
            range_max = self.c + self.V @ domain_max
        except ValueError:
            print("[Linear Star Set]")
            ic(self.V.shape)
            ic(self.c.shape)
            ic(v.shape)
            raise

        return range_max
