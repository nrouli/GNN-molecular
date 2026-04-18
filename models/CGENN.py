import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph, global_mean_pool, global_add_pool

from models.algebra.cliffordalgebra import CliffordAlgebra
from models.ga_modules.gp import SteerableGeometricProductLayer
from models.ga_modules.linear import MVLinear
from models.ga_modules.mvlayernorm import MVLayerNorm
from models.ga_modules.mvsilu import MVSiLU


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class CEMLP(nn.Module):
    """Clifford-Equivariant MLP: MVLinear -> MVSiLU -> GeometricProduct -> LayerNorm."""

    def __init__(
        self,
        algebra,
        in_features,
        hidden_features,
        out_features,
        n_layers=2,
        normalization_init=0,
    ):
        super().__init__()
        self.algebra = algebra
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.n_layers = n_layers

        layers = []
        for i in range(n_layers - 1):
            layers.append(
                nn.Sequential(
                    MVLinear(self.algebra, in_features, hidden_features),
                    MVSiLU(self.algebra, hidden_features),
                    SteerableGeometricProductLayer(
                        self.algebra,
                        hidden_features,
                        normalization_init=normalization_init,
                    ),
                    MVLayerNorm(self.algebra, hidden_features),
                )
            )
            in_features = hidden_features

        layers.append(
            nn.Sequential(
                MVLinear(self.algebra, in_features, out_features),
                MVSiLU(self.algebra, out_features),
                SteerableGeometricProductLayer(
                    self.algebra,
                    out_features,
                    normalization_init=normalization_init,
                ),
                MVLayerNorm(self.algebra, out_features),
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class EGCL(nn.Module):
    """Equivariant Graph Convolutional Layer operating on multivectors."""

    def __init__(
        self,
        algebra,
        in_features,
        hidden_features,
        out_features,
        edge_attr_features=0,
        node_attr_features=0,
        residual=True,
        normalization_init=0,
    ):
        super().__init__()
        self.residual = residual
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.edge_attr_features = edge_attr_features
        self.node_attr_features = node_attr_features

        self.edge_model = CEMLP(
            algebra,
            self.in_features + self.edge_attr_features,
            self.hidden_features,
            self.out_features,
            normalization_init=normalization_init,
        )

        self.node_model = CEMLP(
            algebra,
            self.in_features + self.out_features,
            self.hidden_features,
            self.out_features,
            normalization_init=normalization_init,
        )

    def message(self, h_i, h_j, edge_attr=None):
        if edge_attr is None:
            input = h_i - h_j
        else:
            input = torch.cat([h_i - h_j, edge_attr], dim=1)
        return self.edge_model(input)

    def aggregate(self, h_msg, segment_ids, num_segments):
        return unsorted_segment_mean(h_msg, segment_ids, num_segments=num_segments)

    def update(self, h_agg, h, node_attr=None):
        if node_attr is not None:
            input_h = torch.cat([h, h_agg, node_attr], dim=1)
        else:
            input_h = torch.cat([h, h_agg], dim=1)

        out_h = self.node_model(input_h)

        if self.residual:
            out_h = h + out_h

        return out_h

    def forward(self, h, edge_index, edge_attr=None, node_attr=None):
        rows, cols = edge_index
        h_i, h_j = h[rows], h[cols]

        # Message
        h_msg = self.message(h_i, h_j, edge_attr)

        # Aggregate
        agg_h = self.aggregate(
            h_msg.flatten(1), rows, num_segments=len(h)
        ).view(len(h), *h_msg.shape[1:])

        # Update
        h = self.update(agg_h, h, node_attr)
        return h


class CGGNN(nn.Module):
    """
    Clifford Group Graph Neural Network for molecular property prediction.

    Operates in Cl(3,0,0). Node features are multivectors with:
      - Scalar (grade 0) channels from atom embeddings
      - Vector (grades 1,2,3) channel from mean-centered positions

    Edge features are RBF-expanded interatomic distances embedded as
    scalar multivectors.

    Args:
        max_z: Maximum atomic number supported.
        hidden_features: Number of multivector channels in hidden layers.
        out_features: Number of output scalar targets.
        n_layers: Number of EGCL message-passing layers.
        n_rbf: Number of radial basis functions for distance expansion.
        cutoff: Radius cutoff for building the interaction graph (Angstroms).
        max_neighbors: Maximum neighbors per node in the radius graph.
        normalization_init: Initialization for geometric product normalization.
        residual: Whether to use residual connections in EGCL layers.
        embed_positions: If True, embed centered coordinates as grade-1 features.
        readout: Global pooling type; 'add' for extensive, 'mean' for intensive.
    """

    def __init__(
        self,
        max_z=100,
        hidden_features=64,
        out_features=1,
        n_layers=4,
        n_rbf=20,
        cutoff=10.0,
        normalization_init=0,
        residual=True,
        embed_positions=True,
        readout="add",
    ):
        super().__init__()
        self.algebra = CliffordAlgebra((1.0, 1.0, 1.0))
        self.cutoff = cutoff
        self.n_rbf = n_rbf
        self.readout = readout
        self.embed_positions = embed_positions

        # --- Node embedding ---
        self.atom_embed = nn.Embedding(max_z, hidden_features)

        # Input: hidden scalar channels + (optionally) 1 vector channel from positions
        in_channels = hidden_features + (1 if embed_positions else 0)
        self.input_proj = MVLinear(
            self.algebra, in_channels, hidden_features, subspaces=False
        )

        # --- Edge embedding ---
        # RBF centers (not learnable by default)
        centers = torch.linspace(0.0, cutoff, n_rbf)
        self.register_buffer("rbf_centers", centers)
        self.rbf_width = cutoff / n_rbf

        # --- Message-passing layers ---
        self.layers = nn.ModuleList(
            [
                EGCL(
                    self.algebra,
                    hidden_features,
                    hidden_features,
                    hidden_features,
                    edge_attr_features=n_rbf,
                    residual=residual,
                    normalization_init=normalization_init,
                )
                for _ in range(n_layers)
            ]
        )

        # --- Readout ---
        self.pre_pool = MVLinear(self.algebra, hidden_features, hidden_features)
        self.post_pool = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features),
        )

    def rbf_expansion(self, dist):
        """Gaussian RBF expansion of interatomic distances."""
        # dist: [E] -> [E, n_rbf]
        return torch.exp(
            -((dist.unsqueeze(-1) - self.rbf_centers) ** 2)
            / (2 * self.rbf_width**2)
        )

    def forward(self, data):
        """
        Args:
            data: PyG Data/Batch with at least z, pos, batch.

        Returns:
            Tensor of shape [batch_size, out_features] with predicted properties.
        """

        z, pos, batch = data.z, data.pos, data.batch
        edge_index = data.edge_index  # precomputed by transform

        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)

        # ---- Edge features: RBF of distances -> scalar multivectors ----
        dist = (pos[row] - pos[col]).norm(dim=-1)  # [E]
        rbf = self.rbf_expansion(dist)  # [E, n_rbf]
        # Each RBF channel becomes a scalar (grade-0) multivector
        edge_attr = self.algebra.embed(rbf.unsqueeze(-1), (0,))  # [E, n_rbf, 8]

        # ---- Node features ----
        # Scalar channels from atom type
        h_scalar = self.atom_embed(z)  # [N, hidden]
        h_scalar = self.algebra.embed(h_scalar.unsqueeze(-1), (0,))  # [N, hidden, 8]

        if self.embed_positions:
            # Vector channel from mean-centered coordinates (translation invariant)
           

            pos_mean = global_mean_pool(pos, batch)  # [B, 3]
            pos_centered = pos - pos_mean[batch]  # [N, 3]
            # 3D position -> grade-1 (vector) multivector, 1 channel
            h_vec = self.algebra.embed(
                pos_centered.unsqueeze(1), (1, 2, 3)
            )  # [N, 1, 8]
            h = torch.cat([h_scalar, h_vec], dim=1)  # [N, hidden+1, 8]
        else:
            h = h_scalar  # [N, hidden, 8]

        # Project to uniform hidden dimension
        h = self.input_proj(h)  # [N, hidden, 8]

        # ---- Message passing ----
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr=edge_attr)

        # ---- Readout: extract scalars, pool, predict ----
        h = self.pre_pool(h)  # [N, hidden, 8]
        h = h[..., 0]  # Extract grade-0 (scalar) part -> [N, hidden]

        if self.readout == "add":
            h = global_add_pool(h, batch)  # [B, hidden]
        else:
            h = global_mean_pool(h, batch)  # [B, hidden]

        return self.post_pool(h)  # [B, out_features]