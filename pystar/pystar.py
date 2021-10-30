from __future__ import annotations
from copy import deepcopy
from scipy.optimize import linprog
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt

ColumVector = np.ndarray
RowVector = np.ndarray
Matrix = np.ndarray
ListMatrix = list[list[float]]
ListRowVector = list[list[float]]
ListColumnVector = list[list[float]]

class Plot2D:
    def __init__(self, title, num_directions, domain_x_range=None, domain_y_range=None, range_x_range=None, range_y_range=None):
        self.num_directions = num_directions
        self.title = title
        self.domain_x_range = domain_x_range
        self.domain_y_range = domain_y_range
        self.range_x_range = range_x_range
        self.range_y_range = range_y_range

    def __call__(self, sl):
        dverts = []
        rverts = []
        for s in sl:
            dvert, rvert = vertices(s, self.num_directions)
            if dvert:
                dvert.append(dvert[0])
                dverts.append(dvert)
            if rvert:
                rvert.append(rvert[0])
                rverts.append(rvert)

        if dverts:
            fig = plt.figure()
            fig.suptitle(self.title)
            plt.subplot(1, 2, 1)
            plt.title('Domain')
            for dvert in dverts:
                x = [p[0] for p in dvert]
                y = [p[1] for p in dvert]
                plt.plot(x, y)
            plt.gca().set_aspect('equal', adjustable='box')
            if self.domain_x_range:
                plt.xlim(self.domain_x_range)
            if self.domain_y_range:
                plt.ylim(self.domain_y_range)

            plt.subplot(1, 2, 2)
            plt.title('Range')
            for rvert in rverts:
                x = [p[0] for p in rvert]
                y = [p[1] for p in rvert]
                plt.plot(x, y)

            if self.range_x_range:
                plt.xlim(self.range_x_range)
            if self.range_y_range:
                plt.ylim(self.range_y_range)

            plt.gca().set_aspect('equal', adjustable='box')
        else:
            fig = plt.figure()
            fig.suptitle(self.title)
            plt.title('Range')
            for rvert in rverts:
                x = [p[0] for p in rvert]
                y = [p[1] for p in rvert]
                plt.plot(x, y)

            if self.range_x_range:
                plt.xlim(self.range_x_range)
            if self.range_y_range:
                plt.ylim(self.range_y_range)

            plt.gca().set_aspect('equal', adjustable='box')


        return sl

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, sl):
        for layer in self.layers:
            sl = layer(sl)

        return sl


class Dense:
    def __init__(self: Dense, W: Matrix, b: ColumnVector):
        self.W = W
        self.b = b

    def __call__(self: Dense, sl: list[LinearStarSet]):
        res = []
        for s in sl:
            Vp = self.W @ s.V
            cp = self.W @ s.c + self.b
            ss = LinearStarSet(cp, Vp, s.H)
            res.append(ss)
        return res

class StepReLU:
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, sl):
        dim = self.dim
        slp = []
        for s in sl:
            v = np.zeros((1, s.c.shape[0]))
            v[0, dim] = -1
            res = s @ v
            if res[dim] >= 0:
                slp.append(s)
            else:
                # First case: x_dim == 0
                A_ub = [[0] * s.c.shape[0]]
                b_ub = [[0]]
                A_ub[0][dim] = 1
                s1 = s & HPolytope(A_ub, b_ub) # Restrict domain to x_i <= 0
                # Collapse dimension to 0
                p0 = np.identity(s.c.shape[0])
                p0[dim, dim] = 0
                Vp = p0 @ s.V
                cp = p0 @ s.c
                s1 = LinearStarSet(cp, Vp, s1.H)
                slp.append(s1)

                # Second case: x_dim > 0
                res = s @ -v
                if res[dim] > 0:
                    A_ub = [[0] * s.c.shape[0]]
                    b_ub = [[0]]
                    A_ub[0][dim] = -1
                    s2 = s & HPolytope(A_ub, b_ub)
                    slp.append(s2)

        return slp

class ReLU:
    def __init__(self, overapprox=False, overapprox_dims=1):
        self.overapprox = overapprox
        self.overapprox_dims = overapprox_dims
        if self.overapprox:
            self.process = ReLU._overapprox
        else:
            self.process = ReLU._exact

    @staticmethod
    def _overapprox(s: LinearStarSet):
        for dim in range(s.c.shape[0]):
            v = np.zeros((1, s.c.shape[0]))
            v[0, dim] = -1
            res = s @ v
            l = res[dim]
            if l >= 0:
                continue
            res = s @ -v
            u = res[dim]

            Vi = s.V[dim] # Get the ith-row
            ci = s.c[dim] # Get the ith-row

            p0 = np.identity(s.c.shape[0])
            p0[dim, dim] = 0
            s.V = p0 @ s.V
            s.c = p0 @ s.c

            if u > 0:
                d = u/(u - l)
                ei = np.zeros(s.c.shape[0]).reshape(-1, 1)
                ei[dim] = 1
                s.V = np.append(s.V, ei, axis=1)
                for row in s.H.A_ub:
                    row.append(0.0)

                A_ub = [[0] * Vi.shape[0] + [-1],
                        Vi.tolist() + [-1],
                        (-d*Vi).tolist() + [1]]
                b_ub = [[0], (-ci).tolist(), (d*(ci - l)).tolist()]
                H = HPolytope(A_ub, b_ub)
                s.H &= H

        return [s]

    @staticmethod
    def _exact(s: LinearStarSet):
        slp = [s]
        for dim in range(s.c.shape[0]):
            slp = ReLU._step(slp, dim)

        return slp

    @staticmethod
    def _step(sl: LinearStarSet, dim: int):
        slp = []
        for s in sl:
            v = np.zeros((1, s.c.shape[0]))
            v[0, dim] = -1
            res = s @ v
            if res[dim] >= 0:
                slp.append(s)
            else:
                # First case: x_dim == 0
                A_ub = [[0] * s.c.shape[0]]
                b_ub = [[0]]
                A_ub[0][dim] = 1
                s1 = s & HPolytope(A_ub, b_ub) # Restrict domain to x_i <= 0
                p0 = np.identity(s.c.shape[0])
                p0[dim, dim] = 0
                Vp = p0 @ s.V
                cp = p0 @ s.c
                s1 = LinearStarSet(cp, Vp, s1.H)
                slp.append(s1)

                # Second case: x_dim > 0
                res = s @ -v
                if res[dim] > 0:
                    A_ub = [[0] * s.c.shape[0]]
                    b_ub = [[0]]
                    A_ub[0][dim] = -1
                    s2 = s & HPolytope(A_ub, b_ub)
                    slp.append(s2)

        return slp

    def __call__(self: ReLU, sl: list[LinearStarSet]):
        res = []
        for s in sl:
            slp = self.process(s)
            res.extend(slp)

        return res
            
class HPolytope:
    def __init__(self: HPolytope, A_ub: ListMatrix, b_ub: ListColumnVector) -> None:
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
        self.b_ub.extend(deepcopy(H.b_ub))
        return self

    def __matmul__(self: HPolytope, v: RowVector) -> ColumnVector:
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

        res = res.x.reshape((-1, 1))
        return res

class LinearStarSet:
    def __init__(self: LinearStarSet, c: ColumnVector, V: Matrix, H: HPolytope) -> None:
        self.c = c
        self.V = V
        self.H = H

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

    def __matmul__(self: LinearStarSet, v: RowVector) -> ColumVector:
        # Convert to domain
        try:
            domain_dir = v @ self.V
            domain_max = self.H @ domain_dir
            range_max = self.c + self.V @ domain_max
        except ValueError:
            print("[Linear Star Set]")
            ic(self.V.shape)
            ic(self.c.shape)
            ic(v.shape)
            raise

        return range_max


def column(vec: list[float]) -> ColumnVector:
    return np.array(vec, dtype=float).reshape(-1, 1)

def row(vec: list[float]) -> RowVector:
    return np.array(vec, dtype=float).reshape(1, -1)

def vertices(s: LinearStarSet, num_directions: int=50):
    dverts = []
    rverts = []
    last_d = None
    last_r = None
    if len(s.H.A_ub[0]) == 2:
        for theta in np.linspace(0, 2*np.pi, num_directions):
            v = row([np.cos(theta), np.sin(theta)])
            d = s.H @ v
            r = s.c + s.V @ d
            if last_d is None or not np.allclose(d.flat, last_d.flat):
                last_d = d
                dverts.append(d)
            if last_r is None or not np.allclose(r.flat, last_r.flat):
                last_r = r
                rverts.append(r)
    else:
        for theta in np.linspace(0, 2*np.pi, num_directions):
            v = row([np.cos(theta), np.sin(theta)])
            r = s @ v
            if last_r is None or not np.allclose(r.flat, last_r.flat):
                last_r = r
                rverts.append(r)

    return dverts, rverts
    

if __name__ == '__main__':
    c = column([0, 0])
    V = np.array([
        [np.cos(np.pi/4), -np.sin(np.pi/4)], 
        [np.sin(np.pi/4), np.cos(np.pi/4)]
    ])
    A_ub = [[1.0, 0.0], 
            [0.0, 1.0], 
            [-1.0, 0], 
            [0.0, -1.0], 
            [2.0, 1.0]]
    b_ub = column([1.0, 1.0, 1.0, 1.0, 1.0]).tolist()
    H = HPolytope(A_ub, b_ub)

    f = LinearStarSet(c, V, H)
    plot(f)
    f &= HPolytope([
        [1, 0], 
        [0, -1]], 
        column([1, 0]).tolist())
    plot(f)
    plt.show()
