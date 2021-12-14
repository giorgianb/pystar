from __future__ import annotations
import torch
from pystar.types import LinearStarSetBatch
from pystar.types import LinearStarSet, HPolytope
from itertools import product
from icecream import ic
import torch.functional as F

def compute_maxpool_shape(input_dim, kernel_size, stride, dilation):
    x = (input_dim[0] - dilation[0] * (kernel_size[0] - 1) - 1)//stride[0] + 1
    y = (input_dim[1] - dilation[1] * (kernel_size[1] - 1) - 1)//stride[1] + 1

    return (x, y)

class MaxPool2d(torch.nn.Module):
    def __init__(self: MaxPool2d, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False, overapprox=False):
        # TODO: support the remaining parameters
        super().__init__()
        if return_indices:
            raise NotImplementedError('MaxPool2d does not support returning indices.')
        if ceil_mode:
            raise NotImplementedError('MaxPool2d does not yet support ceil_mode=True')
        if not overapprox:
            raise NotImplementedError('MaxPool2d does not yet support exact analysis.')

        self.kernel_size = kernel_size if type(kernel_size) is tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) is tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation if type(dilation) is tuple else (dilation, dilation)
        self.overapprox = overapprox
        if self.overapprox:
            self.process = self._overapprox
        else:
            self.process = self._exact

    def _get_max_candidates(self: MaxPool2d, s: LinearStarSet):
        all_max_candidates = []
        all_max_values = []
        # number of new predicates we will need to add
        n_new_predicates = 0
        # number of new constraints we will need
        n_new_constraints = 0

        kx, ky = self.kernel_size
        sx, sy = self.stride
        dx, dy = self.dilation

        sp = s.clone()
        for dim in product(*map(range, s.shape[:-2])):
            ci = s.c[dim]
            Vi = s.V[dim]

            channel_max_candidates = []
            channel_max_values = []
            # TODO: deal with padding
            # TODO: this part is extremely inefficient
            # we need a faster way of finding max candidates

            max_i = ci.shape[0] - (kx * dx - 1)
            max_j = ci.shape[1] - (ky * dy - 1)
            
            # i represents the x-coordinate of the top-left corner of the pooling window
            for i in range(0, max_i, sx):
                row_max_candidates = []
                row_max_values = []
                # j represents the x-coordinate of the top-left corner of the pooling window
                for j in range(0, max_j, sy):
                    max_candidates = []
                    max_values = []
                    # (ki, kj) is our max candidate
                    for ki in range(0, kx * dx, dx):
                        for kj in range(0, ky * dy, dy):
                            # Make a clone of the constraints
                            H = s.H.clone()

                            # (kii, kjj) are the variables that our max candidate
                            # has to be greater than
                            for kii in range(0, kx * dx, dx):
                                for kjj in range(0, ky * dy, dy):
                                    if (ki, kj) == (kii, kjj):
                                        continue
                                    # TODO: replace with append, and stack
                                    # all of the new constraints at the end

                                    # Our max-candidate has to be greater than this variable
                                    A_ub = (Vi[i + kii, j + kjj] - Vi[i + ki, j + kj]).type(
                                            H.A_ub.dtype
                                    )

                                    b_ub = torch.tensor([
                                        ci[i + ki, j + kj] - ci[i + kii, j + kjj]
                                        ], dtype=H.b_ub.dtype)
                                    H &= HPolytope(torch.unsqueeze(A_ub, 0), b_ub)

                            sp.H = H
                            v = torch.zeros(s.shape)
                            v[dim + (i + ki, j + kj)] = 1
                            res = sp.maximize(v)
                            if res is not None:
                                # remember the current pixel as a max candidate
                                max_candidates.append((i + ki, j + kj))
                                max_values.append(res[dim + (i + ki, j + kj)])

                    row_max_candidates.append(max_candidates)
                    row_max_values.append(max_values)

                    assert len(max_candidates) >= 1
                    if len(max_candidates) > 1:
                        n_new_predicates += 1
                        n_new_constraints += len(max_candidates) + 1
                channel_max_candidates.append(row_max_candidates)
                channel_max_values.append(row_max_values)

            all_max_candidates.append(channel_max_candidates)
            all_max_values.append(channel_max_values)

            return all_max_candidates, all_max_values, n_new_predicates, n_new_constraints

    def _overapprox(self: MaxPool2d, s: LinearStarSet):
        padding = self.padding
        if padding > 0:
            n_rows, n_columns, n_generators = s.V.shape[-3:]
            # pad anchor below
            p = -torch.inf * torch.ones(s.shape[:-2] + (padding, n_columns), dtype=s.dtype)
            c = torch.cat((p1, s.c, p1), axis=-2)

            # pad anchor above
            p = torch.ones(s.shape[:-2], + (n_rows + 2*padding, padding), dtype=s.dtype)
            c = torch.cat((p2, c, p2), axis=-1)

            # pad generators below
            p = torch.ones(s.V.shape[:-3] + (padding, n_columns, n_generators), dtype=s.dtype)
            V = torch.cat((p, s.V, p), axis=-3)
            # pad generators above
            p = torch.ones(
                    s.V.shape[:-3] + (n_rows + 2*padding, padding, n_generator), 
                    dtype=s.dtype
            )
            V = torch.cat((p, s.V, p), axis=-2)
            s = LinearStarSet(c, V, s.H.clone())

        all_max_candidates, all_max_values, n_new_predicates, n_new_constraints = self._get_max_candidates(s)
        
        wp, hp = compute_maxpool_shape(s.shape[:2], self.kernel_size, self.stride, self.dilation)
        n_old_predicates = s.V.shape[-1]

        # new anchor image
        cp = torch.zeros(s.shape[:-2] + (wp, hp), dtype=s.c.dtype)

        # generators that follow the old constraints
        Vp = torch.zeros(s.shape[:-2] + (wp, hp, n_old_predicates), dtype=s.V.dtype)

        # generators for which we add new constraints
        Vpp = torch.zeros(s.shape[:-2] + (wp, hp, n_new_predicates), dtype=s.V.dtype)

        # new predicates added for max-pooling
        A_ub = torch.zeros((n_new_constraints, n_new_predicates + s.H.A_ub.shape[1]),
                dtype=s.H.A_ub.dtype)
        b_ub = torch.zeros(n_new_constraints, dtype=s.H.b_ub.dtype)

        p_counter = 0 # index of unused predicate
        c_counter = 0 # index of unused constraint row

        max_i = s.shape[-2] - (self.kernel_size[0] * self.dilation[0] - 1)
        max_j = s.shape[-1] - (self.kernel_size[1] * self.dilation[1] - 1)

        kx, ky = self.kernel_size
        sx, sy = self.stride
        dx, dy = self.dilation

        for i, dim in enumerate(product(*map(range, s.shape[:-2]))):
            ci = s.c[dim]
            Vi = s.V[dim]

            mci = all_max_candidates[i]
            mvi = all_max_values[i]

            # pi is row coordinate in pooled image
            # i is row coordinate in unpooled image
            for pi, i in enumerate(range(0, max_i, sx)):
                # pj is column coordinate in pooled image
                # j is column coordinate in unpooled image
                for pj, j in enumerate(range(0, max_j, sy)):
                    max_candidates = mci[pi][pj]
                    max_values = mvi[pi][pj]
                    # If we only have a single max candidate, the 
                    # value is exactly the same
                    if len(max_candidates) == 1:
                        xp, yp = max_candidates[0]
                        Vp[dim + (pi, pj)] = Vi[xp, yp]
                        cp[dim + (pi, pj)] = ci[xp, yp]
                    else:
                        # If we have multiple max candidates
                        # create a new generator which we make
                        # greater than all of the max canidates
                        p = p_counter
                        p_counter += 1
                        Vp[dim + (pi, pj, p)] = 1
                        # We live cp[(dim + (pi, pj)] as zero
                        # as we want the ne generator to fully
                        # control the value 

                        c = c_counter
                        c_counter += len(max_candidates) + 1
                        for pred_i, (xp, yp) in enumerate(max_candidates):
                            A_ub[c + pred_i, c] = -1
                            A_ub[c + pred_i, n_new_predicates:] = Vi[xp, yp]
                            b_ub[c + pred_i] = -ci[xp, yp]
                            assert c  < n_new_predicates

                        A_ub[c + len(max_candidates), c] = 1
                        b_ub[c + len(max_candidates)] = max(max_values)
                        

        pad = torch.zeros((s.H.A_ub.shape[0], n_new_predicates), dtype=s.H.A_ub.dtype)
        A_ub_p = torch.cat((pad, s.H.A_ub), axis=1)
        A_ub_full = torch.cat((A_ub_p, A_ub), axis=0)
        b_ub_full = torch.cat((s.H.b_ub, b_ub), axis=0)
        H = HPolytope(A_ub_full, b_ub_full)
        # Vp must go before Vpp to line up with the way we
        # aded the constraints for the new predicate variables
        V_full = torch.cat((Vp, Vpp), axis=-1)
        s_full = LinearStarSet(cp, V_full, H)

        return [s_full]

    def _linear_star_set_forward(self, input: LinearStarSetBatch):
        res = []
        for s in input:
            ss = self.process(s)
            res.extend(ss)
        return res

    def _tensor_forward(self, input: torch.Tensor):
        # We still don't support ceil_mode
        # and indices yet
        # We can support indices just by keeping track of the max_candidates
        # and returning all of them
        return F.max_pool2d(input, self.kernel_size, self.stride, self.padding, self.dilation, False, False)

    def forward(self, input: Union[LinearStarSetBatch, torch.Tensor]):
        if type(input) == torch.Tensor:
            return self._tensor_forward(input)
        elif type(input) == list:
            return self._linear_star_set_forward(input)
        else:
            raise TypeError("Input must be either torch.Tensor or pystar.types.LinearStarSetBatch")
