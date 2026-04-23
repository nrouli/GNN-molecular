import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool


class GaussianRBF(nn.Module):
    def __init__(self, num_rbf=20, cutoff=10.0):
        super().__init__()
        offsets = torch.linspace(0, cutoff, num_rbf)
        self.register_buffer('offsets', offsets)
        self.width = (offsets[1] - offsets[0]).item()

    def forward(self, dist):
        return torch.exp(-0.5 * ((dist.unsqueeze(-1) - self.offsets) / self.width) ** 2)


class GAT(nn.Module):
    def __init__(
        self,
        node_input_dim=11,
        hidden_dim=128,
        out_dim=1,
        num_layers=4,
        num_heads=4,
        num_rbf=20,
        cutoff=10.0,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = hidden_dim // num_heads

        self.rbf = GaussianRBF(num_rbf, cutoff)

        self.node_embed = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Project RBF edge features to match GATv2's edge_attr expectation
        self.edge_embed = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.conv_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=self.head_dim,
                    heads=num_heads,
                    concat=True,           # output is head_dim * num_heads = hidden_dim
                    edge_dim=hidden_dim,    # accepts edge features
                    add_self_loops=False,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, data):
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch

        h = self.node_embed(x)

        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)
        e = self.rbf(dist)
        e = self.edge_embed(e)

        for conv, norm in zip(self.conv_layers, self.norms):
            h_res = h
            h = conv(h, edge_index, edge_attr=e)
            h = F.silu(h)
            h = h + h_res
            h = norm(h)

        h = global_add_pool(h, batch)
        out = self.output_mlp(h)
        return out.squeeze(-1)