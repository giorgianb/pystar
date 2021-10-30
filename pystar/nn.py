from __future__ import annotations
import torch
from typing import Union
from pystar.pystar import LinearStarSet

class Linear(torch.nn.Module):
    def __init__(in_features: int, out_feautures: int, bias: bool = True, device=None, dtype=None):
        self.W = torch.nn.Parameter(torch.randn(in_features, out_features, device=device))

        self.bias = bias
        if bias:
            self.b = torch.nn.Parameter(torch.randn(out_features, device=device))

    def _tensor_forward(self, input: torch.Tensor):
        ret = input @ self.W
        if self.bias:
            ret += self.b

        return ret

    def _linear_star_set_forward(self, input: list[LinearStarSet]):
        res = []
        for s in sl:
            Vp = self.V @ self.W
            cp = s.c @ self.W + self.b
            ss = LinearStarSet(cp, Vp, s.H)
            res.append(ss)
        return res

    def forward(self, input: Union[list[LinearStarSet], torch.Tensor]):
        if type(input) == torch.Tensor:
            return self._tensor_forward(input)
        elif type(input) == list:
            return self._linear_star_set_forward(self, input)

