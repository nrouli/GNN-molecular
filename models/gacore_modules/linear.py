import math

import torch
from torch import nn

from .utils import unsqueeze_like


class MVLinear(nn.Module):
    def __init__(
        self,
        algebra,
        in_features,
        out_features,
        subspaces=True,
        bias=True,
    ):
        super().__init__()

        self.algebra = algebra
        self.in_features = in_features
        self.out_features = out_features
        self.subspaces = subspaces

        if subspaces:
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features, algebra.n_subspaces)
            )
            
            self.register_buffer(
                "_expand_idx",
                torch.arange(algebra.n_subspaces).repeat_interleave(algebra.subspaces),
            )
            self._forward = self._forward_subspaces
        else:
            self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(1, out_features, 1))
            self.b_dims = (0,)
        else:
            self.register_parameter("bias", None)
            self.b_dims = ()

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def _forward(self, input):
        return torch.einsum("bm...i, nm->bn...i", input, self.weight)

    def _forward_subspaces(self, input):
        weight = self.weight[..., self._expand_idx]
        return torch.einsum("bm...i, nmi->bn...i", input, weight)

    def forward(self, input):
        result = self._forward(input)

        if self.bias is not None:
            bias = self.algebra.embed(self.bias, self.b_dims)
            result += unsqueeze_like(bias, result, dim=2)
        return result
