import math

import torch
from torch import nn

from .linear import MVLinear
from .normalization import NormalizationLayer


class FullyConnectedSteerableGeometricProductLayer(nn.Module):
    def __init__(
        self,
        algebra,
        in_features,
        out_features,
        include_first_order=True,
        normalization_init=0,
    ):
        super().__init__()

        self.algebra = algebra
        self.in_features = in_features
        self.out_features = out_features
        self.include_first_order = include_first_order

        if normalization_init is not None:
            self.normalization = NormalizationLayer(
                algebra, in_features, normalization_init
            )
        else:
            self.normalization = nn.Identity()
        self.linear_right = MVLinear(algebra, in_features, in_features, bias=False)
        if include_first_order:
            self.linear_left = MVLinear(algebra, in_features, out_features, bias=True)

        self.product_paths = algebra.geometric_product_paths
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, self.product_paths.sum())
        )

        self.reset_parameters()

        # ── Precompute grade-to-blade expansion mapping ──
        subspaces = algebra.subspaces
        n_blades = int(subspaces.sum())
        cayley = algebra.cayley

        grade_indices = self.product_paths.nonzero(as_tuple=False)

        starts = torch.zeros_like(subspaces)
        starts[1:] = subspaces[:-1].cumsum(0)

        blade_positions = []
        param_indices = []
        cayley_values = []

        for p_idx, (gi, gj, gk) in enumerate(grade_indices):
            gi, gj, gk = gi.item(), gj.item(), gk.item()
            for bi in range(starts[gi], starts[gi] + subspaces[gi]):
                for bj in range(starts[gj], starts[gj] + subspaces[gj]):
                    for bk in range(starts[gk], starts[gk] + subspaces[gk]):
                        c = cayley[bi, bj, bk].item()
                        if c != 0:
                            blade_positions.append((bi, bj, bk))
                            param_indices.append(p_idx)
                            cayley_values.append(c)

        if len(blade_positions) > 0:
            bp = torch.tensor(blade_positions, dtype=torch.long)
            self.register_buffer("_blade_i", bp[:, 0])
            self.register_buffer("_blade_j", bp[:, 1])
            self.register_buffer("_blade_k", bp[:, 2])
            self.register_buffer(
                "_param_idx", torch.tensor(param_indices, dtype=torch.long)
            )
            self.register_buffer(
                "_cayley_vals", torch.tensor(cayley_values, dtype=torch.float)
            )
        else:
            self.register_buffer("_blade_i", torch.zeros(0, dtype=torch.long))
            self.register_buffer("_blade_j", torch.zeros(0, dtype=torch.long))
            self.register_buffer("_blade_k", torch.zeros(0, dtype=torch.long))
            self.register_buffer("_param_idx", torch.zeros(0, dtype=torch.long))
            self.register_buffer("_cayley_vals", torch.zeros(0))

        self._n_blades = n_blades

    def reset_parameters(self):
        torch.nn.init.normal_(
            self.weight,
            std=1 / math.sqrt(self.in_features * (self.algebra.dim + 1)),
        )

    def _get_weight(self):
        n = self._n_blades
        weight = self.weight.new_zeros(self.out_features, self.in_features, n, n, n)
        w = self.weight[:, :, self._param_idx] * self._cayley_vals
        weight[:, :, self._blade_i, self._blade_j, self._blade_k] = w
        return weight

    def forward(self, input):
        input_right = self.linear_right(input)
        input_right = self.normalization(input_right)

        weight = self._get_weight()

        tmp = torch.einsum("mnijk, bnk -> bmnij", weight, input_right)
        bilinear = torch.einsum("bni, bmnij -> bmj", input, tmp)

        if self.include_first_order:
            return (self.linear_left(input) + bilinear) / math.sqrt(2)
        else:
            return bilinear
