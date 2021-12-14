from __future__ import annotations
import torch
from pystar.types import LinearStarSetBatch
from pystar.types import LinearStarSet, HPolytope
from itertools import product

import numpy as np
from scipy.optimize import newton

from icecream import ic

class Sigmoid(torch.nn.Module):
    def __init__(self: Sigmoid):
        super().__init__()
        self.process = self._overapprox

    def _overapprox(self: Sigmoid, s: LinearStarSet):
        s0 = s
        s = LinearStarSet(s.c.clone(), s.V.clone(), s.H.clone())
        for dim in product(*map(range, s.shape)):
            v = torch.zeros(s.shape, dtype=s.dtype)
            v[dim] = -1
            res = s0.maximize(v)
            l = res[dim]
            res = s0.maximize(-v)
            u = res[dim]

            Vi = s.V[dim].clone() # Get the generator for the dim
            ci = s.c[dim].clone() # Get the value for dim

            s.V[dim] = 0
            s.c[dim] = 0

            # Create a new generator to control the value of the variable
            ei = torch.zeros(s.shape, dtype=s.dtype)
            ei[dim] = 1
            s.V = torch.cat((s.V, ei.unsqueeze(-1)), -1)
            pad = torch.zeros(s.H.A_ub.shape[0], 1)
            s.H.A_ub = torch.cat((s.H.A_ub, pad), axis=1)

            sl = sigmoid(l)
            su = sigmoid(u)
            dsl = dsigmoid(l)
            dsu = dsigmoid(u)

            A_ub = torch.zeros(4, s.H.A_ub.shape[1], dtype=s.H.A_ub.dtype)
            if l <= 0 and u <= 0:
                # First contraints: y_i <= the line connecting (l, s(l)), to (u, s(u))
                m = (sigmoid(u)-sigmoid(l))/(u - l)
                A_ub[0, :-1] = -m*Vi
                A_ub[0, -1] = 1

            
                # Second constraint: y_i >= tangent line at (l, sl)
                A_ub[1, :-1] = dsl*Vi
                A_ub[1, -1] = -1

                # Third Constraint: y_i >= tangent line at (u, su)
                A_ub[2, :-1] = dsu*Vi
                A_ub[2, -1] = -1

                # Fourth Constraint: y_i >= tangent line at (c, sc)
                A_ub[3, :-1] = m*Vi
                A_ub[3, -1] = -1

                c = find_point_of_average_derivative(l, u)

                b_ub = torch.tensor([
                    (m*(ci - l) + sl), 
                    -(dsl*(ci - l) + sl), 
                    -(dsu*(ci - u) + su),
                    -(m*(ci - c) + sigmoid(c))
                    ], dtype=s.H.b_ub.dtype)
                H = HPolytope(A_ub, b_ub)
                s.H &= H
            elif l >= 0 and u >= 0:
                # First contraints: y_i >= the line connecting (l, s(l)), to (u, s(u))
                m = (su-sl)/(u - l)
                A_ub[0, :-1] = m*Vi
                A_ub[0, -1] = -1

            
                # Second constraint: y_i <= tangent line at (l, sl)
                A_ub[1, :-1] = -dsl*Vi
                A_ub[1, -1] = 1

                # Third Constraint: y_i <= tangent line at (u, su)
                A_ub[2, :-1] = -dsu*Vi
                A_ub[2, -1] = 1

                # Fourth Constraint: y_i <= tangent line at (c, sc)
                A_ub[3, :-1] = -m*Vi
                A_ub[3, -1] = 1

                c = find_point_of_average_derivative(l, u)

                b_ub = torch.tensor([
                    -(m*(ci - l) + sl), 
                    (dsl*(ci - l) + sl), 
                    (dsu*(ci - u) + su),
                    (m*(ci - c) + sigmoid(c))
                    ], dtype=s.H.b_ub.dtype)
                H = HPolytope(A_ub, b_ub)
                s.H &= H
            elif l <= 0 and u >= 0:
                # First constraint: y_i >= tangent line at (l, sl)
                A_ub[0, :-1] = dsl*Vi
                A_ub[0, -1] = -1

                # Second Constraint: y_i <= tangent line at (u, su)
                A_ub[1, :-1] = -dsu*Vi
                A_ub[1, -1] = 1

                # Third Constraint: y_i <= line tangent to curve that starts at (l, sl)
                m1 = get_tight_line_slope(l)
                A_ub[2, :-1] = -m1 * Vi
                A_ub[2, -1] = 1

                # Fourth Constraint: y_i >= line_tangent to curve that starts at (u, su)
                m2 = get_tight_line_slope(u)
                A_ub[3, :-1] = m2 * Vi
                A_ub[3, -1] = -1

                b_ub = torch.tensor([
                    -(dsl*(ci - l) + sl), 
                    dsu*(ci - u) + su,
                    m1*(ci - l) + sl,
                    -(m2*(ci - u) + su)
                    ], dtype=s.H.b_ub.dtype)
                H = HPolytope(A_ub, b_ub)
                s.H &= H

        return [s]

    def _tensor_forward(self, input: torch.Tensor):
        return 1/(1 + torch.exp(-input))

    def _linear_star_set_forward(self, input: LinearStarSetBatch):
        res = []
        for s in input:
            ss = self.process(s)
            res.extend(ss)
        return res

    def forward(self, input: Union[LinearStarSetBatch, torch.Tensor]):
        if type(input) == torch.Tensor:
            return self._tensor_forward(input)
        elif type(input) == list:
            return self._linear_star_set_forward(input)
        else:
            raise TypeError("Input must be either torch.Tensor or pystar.types.LinearStarSetBatch")


def get_tight_line_slope(b):
    """Gets the slope of the line that will most tightly bound the sigmoid, assuming
    that the lower bound is < 0 and the upper bound > 0"""
    f = f0(b)
    fp = f1(b)
    fpp = f2(b)

    sb = sigmoid(b)
    if b >= 0:
        roots = np.roots([
                    sb,
                    2*sb - b - 2,
                    sb - 1
                    ]).real

        roots = roots[roots > 0]
        x0 = np.min(-np.log(roots))

    else: 
        roots = np.roots([
                    sb,
                    2*sb - b + 2,
                    sb - 1
                    ]).real

        roots = roots[roots > 0]
        x0 = np.max(-np.log(roots))


    roots = newton(f, fprime=fp, fprime2=fpp, x0=x0)
    return dsigmoid(roots)




def find_point_of_average_derivative(x1, x2):
    y1 = sigmoid(x1)
    y2 = sigmoid(x2)

    slope = (y2 - y1)/(x2 - x1)
    r = np.roots([slope, 2*slope - 1, slope]).real
    r = np.log(r[r > 0])
    return r[np.logical_and(x1.detach().numpy() <= r, r <= x2.detach().numpy())][0]


# Sigmoid function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Derivative of sigmoid function
def dsigmoid(x):
    return np.exp(-x)/(1 + np.exp(-x))**2

# Sigmoid Tangent Line Function
def f0(l):
    sl = sigmoid(l)
    b = 2*sl - l - 1
    c = sl - 1
    def f(x):
        return sl*np.exp(-2*x) + (x + b)*np.exp(-x) + c

    return f

# First derivative of Sigmoid Tangent Line Function
def f1(l):
    sl = sigmoid(l)
    b = 2*sl - l - 2
    def f(x):
        return -2*sl*np.exp(-2*x) - (x + b)*np.exp(-x)

    return f

# Second derivative of Sigmoid Tangent Line Function
def f2(l):
    sl = sigmoid(l)
    b = (2*sl - l - 3)
    def f(x):
        return 4*sl*np.exp(-2*x) + (x + b)*np.exp(-x)

    return f
