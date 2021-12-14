from __future__ import annotations
import torch
from typing import Union
from pystar.types import LinearStarSet, LinearStarSetBatch

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn((out_features, in_features), device=device, dtype=dtype))

        self.bias = bias
        if bias:
            self.b = torch.nn.Parameter(torch.randn(out_features, device=device, dtype=dtype))

    def _tensor_forward(self, input: torch.Tensor):
        ret = input @ self.W.T
        if self.bias:
            ret += self.b

        return ret

    def _linear_star_set_forward(self, input: LinearStarSetBatch):
        res = []
        for s in input:
            Vp = self.W @ s.V
            cp = s.c @ self.W.T 
            if self.bias:
                cp += self.b
            ss = LinearStarSet(cp, Vp, s.H)
            res.append(ss)
        return res

    def forward(self, input: Union[LinearStarSetBatch, torch.Tensor]):
        if type(input) == torch.Tensor:
            return self._tensor_forward(input)
        elif type(input) == list:
            return self._linear_star_set_forward(input)
        else:
            raise TypeError("Input must be either torch.Tensor or pystar.types.LinearStarSetBatch")
