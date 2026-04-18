import torch
from torch import nn

EPS = 1e-6


class NormalizationLayer(nn.Module):
    def __init__(self, algebra, features, init: float = 0):
        super().__init__()
        self.algebra = algebra
        self.in_features = features

        self.a = nn.Parameter(torch.zeros(self.in_features, algebra.n_subspaces) + init)

        # Precompute expansion index
        self.register_buffer(
            "_expand_idx",
            torch.arange(algebra.n_subspaces).repeat_interleave(algebra.subspaces),
        )

    def forward(self, input):
        assert input.shape[1] == self.in_features

        norms = torch.cat(self.algebra.norms(input), dim=-1)
        s_a = torch.sigmoid(self.a)
        norms = s_a * (norms - 1) + 1
        norms = norms[..., self._expand_idx]
        normalized = input / (norms + EPS)

        return normalized
