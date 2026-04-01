"""
Non-Equivariant GAT Baseline for QM9
=====================================
CGConv-based message passing with:
  - 3D coordinates concatenated as node features (breaks equivariance)
  - Gaussian RBF-expanded pairwise distances as edge features
  - Fully connected molecular graph (all atoms interact)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv, global_add_pool


# ─── Gaussian RBF distance expansion ───────────────────────────────
class GaussianRBF(nn.Module):
    """Expand scalar distances into a basis of Gaussians."""
    def __init__(self, num_rbf=20, cutoff=10.0):
        super().__init__()
        offsets = torch.linspace(0, cutoff, num_rbf)
        self.register_buffer('offsets', offsets)
        self.width = (offsets[1] - offsets[0]).item()

    def forward(self, dist):
        # dist: (E,) -> (E, num_rbf)
        return torch.exp(-0.5 * ((dist.unsqueeze(-1) - self.offsets) / self.width) ** 2)


# ─── Model ──────────────────────────────────────────────────────────
class GAT(nn.Module):
    """
    Architecture:
        Node input: atom features (11D) || 3D coords -> Linear -> h_0
        Edge input: ||r_ij|| -> GaussianRBF -> Linear -> e_ij
        Message passing: L layers of CGConv with edge features, residuals, LayerNorm
        Readout: sum pooling -> MLP -> scalar prediction
    """
    def __init__(
        self,
        node_input_dim=11 + 3,   # 11 atom features + 3 coordinates
        hidden_dim=128,
        out_dim=1,
        num_layers=4,
        num_heads=4,            # unused with CGConv, kept for API compatibility
        num_rbf=20,
        cutoff=10.0,
    ):
        super().__init__()

        # Distance expansion
        self.rbf = GaussianRBF(num_rbf, cutoff)

        # Input embeddings
        self.node_embed = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_embed = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # CGConv layers + norms
        self.conv_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(CGConv(hidden_dim, dim=hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, data):
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch

        # ── Node features ──
        h = torch.cat([x, pos], dim=-1)        # atom features + coords
        h = self.node_embed(h)                 # (N, hidden_dim)

        # ── Edge features: RBF-expanded distances ──
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)  # (E,)
        e = self.rbf(dist)                          # (E, num_rbf)
        e = self.edge_embed(e)                      # (E, hidden_dim)

        # ── Message passing ──
        for conv, norm in zip(self.conv_layers, self.norms):
            h_res = h
            h = conv(h, edge_index, edge_attr=e)
            h = F.silu(h)
            h = h + h_res
            h = norm(h)

        # ── Readout ──
        h = global_add_pool(h, batch)          # (B, hidden_dim)
        out = self.output_mlp(h)               # (B, out_dim)
        return out.squeeze(-1)
