import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_scatter import scatter_add, scatter_max


from models.algebra.cliffordalgebra import CliffordAlgebra
from models.ga_modules.linear import MVLinear
from models.ga_modules.mvlayernorm import MVLayerNorm
from models.ga_modules.mvsilu import MVSiLU



class GA_GATLayer(nn.Module):
    """Geometric Algebra Graph Attention Layer.
 
    Uses grade-aware scoring: each grade contributes independently
    to the scalar attention logit via learned per-grade contraction
    vectors and importance weights.
    """
 
    def __init__(
        self,
        algebra,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.algebra = algebra
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
 
        # Algebra structure
        self.n_grades = algebra.n_subspaces              # 4 for Cl(3,0)
        self.grade_dims = algebra.subspaces.tolist()     # [1, 3, 3, 1]
        self.n_blades = algebra.n_blades                 # 8 for Cl(3,0)
        self.grade_slices = algebra.grade_to_slice       # list of slice objects
 
        # Step 1: Multivector linear projection
        self.proj = MVLinear(
            algebra=algebra,
            in_features=in_channels,
            out_features=out_channels * heads,
            subspaces=True,
        )
 
        # Step 2: Grade-aware attention vectors (source and target)
        self.a_src = nn.ParameterList([
            nn.Parameter(torch.empty(heads, out_channels * d_g))
            for d_g in self.grade_dims
        ])
        self.a_dst = nn.ParameterList([
            nn.Parameter(torch.empty(heads, out_channels * d_g))
            for d_g in self.grade_dims
        ])
 
        # Learned grade importance weights
        self.w_src = nn.Parameter(torch.ones(heads, self.n_grades))
        self.w_dst = nn.Parameter(torch.ones(heads, self.n_grades))
 
        self.leaky_relu = nn.LeakyReLU(negative_slope)
 
        self._reset_parameters()
 
    def _reset_parameters(self):
        for p in self.a_src:
            nn.init.xavier_uniform_(p.unsqueeze(0))  # xavier needs >= 2D
        for p in self.a_dst:
            nn.init.xavier_uniform_(p.unsqueeze(0))
 
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_scalar: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:            (N, in_channels, n_blades) multivector node features
            edge_index:   (2, E) source, target pairs
            edge_scalar:  (E, heads) optional scalar edge bias (from RBF)
 
        Returns:
            (N, heads * out_channels, n_blades) if concat
            (N, out_channels, n_blades)          if average
        """
        src, dst = edge_index
        N = x.size(0)
 
        # Step 1: MVLinear projection -> (N, heads * out_channels, n_blades)
        z = self.proj(x)
        # Reshape to (N, heads, out_channels, n_blades)
        z = z.view(N, self.heads, self.out_channels, self.n_blades)
 
        # Step 2: Grade-aware attention scoring
        score_src = torch.zeros(N, self.heads, device=x.device)
        score_dst = torch.zeros(N, self.heads, device=x.device)
 
        for g, s in enumerate(self.grade_slices):
            # Extract grade g: (N, H, F', d_g) -> flatten to (N, H, F' * d_g)
            z_g = z[..., s].flatten(-2, -1)  # (N, H, F' * d_g)
 
            # Contract to scalar per node per head
            s_src = (z_g * self.a_src[g]).sum(dim=-1)  # (N, H)
            s_dst = (z_g * self.a_dst[g]).sum(dim=-1)  # (N, H)
 
            score_src += self.w_src[:, g] * s_src
            score_dst += self.w_dst[:, g] * s_dst
 
        # Per-edge logit with optional distance bias
        e = score_src[src] + score_dst[dst]  # (E, H)
        if edge_scalar is not None:
            e = e + edge_scalar
        e = self.leaky_relu(e)
 
        # Step 3: Sparse softmax (per destination node neighborhood)
        e_max = scatter_max(e, dst, dim=0, dim_size=N)[0]  # (N, H)
        e_stable = e - e_max[dst]
        alpha = torch.exp(e_stable)  # (E, H)
        alpha_sum = scatter_add(alpha, dst, dim=0, dim_size=N)  # (N, H)
        alpha = alpha / (alpha_sum[dst] + 1e-16)  # (E, H)
 
        # Dropout on attention weights
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
 
        # Step 4: Weighted aggregation
        # alpha: (E, H) -> (E, H, 1, 1) to broadcast over (E, H, F', n_blades)
        out = z[src] * alpha.unsqueeze(-1).unsqueeze(-1)
        out = scatter_add(out, dst, dim=0, dim_size=N)  # (N, H, F', n_blades)
 
        # Step 5: Concat or average heads
        if self.concat:
            # (N, H, F', n_blades) -> (N, H*F', n_blades)
            return out.reshape(N, self.heads * self.out_channels, self.n_blades)
        else:
            # (N, H, F', n_blades) -> (N, F', n_blades)
            return out.mean(dim=1)
 
 
# ─────────────────────────────────────────────────────────────────────────────
 
 
class GaussianRBF(nn.Module):
    """Fixed Gaussian radial basis function expansion."""
 
    def __init__(self, num_rbf=20, cutoff=5.0):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer("centers", centers)
        self.width = (centers[1] - centers[0]).item()
 
    def forward(self, dist):
        return torch.exp(
            -0.5 * ((dist.unsqueeze(-1) - self.centers) / self.width) ** 2
        )
 
 
# ─────────────────────────────────────────────────────────────────────────────
 
 
class GA_GAT(nn.Module):
    """Geometric Algebra Graph Attention Network for molecular property prediction.
 
    Operates in Cl(3,0). Node features are multivectors with:
      - Scalar (grade 0) channels from atom embeddings
      - Vector (grade 1) channel from mean-centered positions
 
    Edge features are RBF-expanded interatomic distances projected
    to scalar attention biases.
    """
 
    def __init__(
        self,
        max_z=100,
        hidden_dim=64,
        out_dim=1,
        num_layers=4,
        num_heads=4,
        num_rbf=20,
        cutoff=10.0,
        dropout=0.0,
        negative_slope=0.2,
        embed_positions=True,
        readout="add",
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = hidden_dim // num_heads
 
        self.algebra = CliffordAlgebra((1.0, 1.0, 1.0))
        self.cutoff = cutoff
        self.readout = readout
        self.embed_positions = embed_positions
        self.num_heads = num_heads
 
        # --- Node embedding ---
        self.atom_embed = nn.Embedding(max_z, hidden_dim)
 
        in_channels = hidden_dim + (1 if embed_positions else 0)
        self.input_proj = MVLinear(
            self.algebra, in_channels, hidden_dim, subspaces=False
        )
 
        # --- Edge embedding: RBF -> scalar bias per head ---
        self.rbf = GaussianRBF(num_rbf, cutoff)
        self.edge_mlp = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_heads),
        )
 
        # --- Attention layers with normalization ---
        self.att_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()
 
        for _ in range(num_layers):
            self.att_layers.append(
                GA_GATLayer(
                    algebra=self.algebra,
                    in_channels=hidden_dim,
                    out_channels=self.head_dim,
                    heads=num_heads,
                    concat=True,  # head_dim * num_heads = hidden_dim
                    dropout=dropout,
                    negative_slope=negative_slope,
                )
            )
            self.norms.append(MVLayerNorm(self.algebra, channels=hidden_dim))
            self.activations.append(MVSiLU(self.algebra, hidden_dim))
 
        # --- Readout ---
        self.pre_pool = MVLinear(self.algebra, hidden_dim, hidden_dim)
        self.post_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
 
    def forward(self, data):
        z, pos, batch = data.z, data.pos, data.batch
        edge_index = data.edge_index
 
        row, col = edge_index
 
        # ---- Edge features: scalar RBF -> attention bias per head ----
        dist = (pos[row] - pos[col]).norm(dim=-1)  # [E]
        rbf = self.rbf(dist)                        # [E, num_rbf]
        edge_scalar = self.edge_mlp(rbf)            # [E, num_heads]
 
        # ---- Node features ----
        h_scalar = self.atom_embed(z)                                     # [N, hidden]
        h_scalar = self.algebra.embed(h_scalar.unsqueeze(-1), (0,))       # [N, hidden, 8]
 
        if self.embed_positions:
            pos_mean = global_mean_pool(pos, batch)                       # [B, 3]
            pos_centered = pos - pos_mean[batch]                          # [N, 3]
            h_vec = self.algebra.embed(
                pos_centered.unsqueeze(1), (1, 2, 3)
            )                                                             # [N, 1, 8]
            h = torch.cat([h_scalar, h_vec], dim=1)                       # [N, hidden+1, 8]
        else:
            h = h_scalar                                                  # [N, hidden, 8]
 
        h = self.input_proj(h)                                            # [N, hidden, 8]
 
        # ---- Message passing ----
        for layer, norm, act in zip(self.att_layers, self.norms, self.activations):
            h_res = h
            h = layer(h, edge_index, edge_scalar=edge_scalar)
            h = act(h)
            h = h + h_res
            h = norm(h)
 
        # ---- Readout: extract scalars, pool, predict ----
        h = self.pre_pool(h)                                              # [N, hidden, 8]
        h = h[..., 0]                                                     # [N, hidden]
 
        if self.readout == "add":
            h = global_add_pool(h, batch)
        else:
            h = global_mean_pool(h, batch)
 
        return self.post_pool(h).squeeze(-1)