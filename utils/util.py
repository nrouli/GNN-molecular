import os
import sys
import copy
import torch
import torch.nn as nn
import numpy as np
import random

from torch_geometric.datasets import QM9
import torch_geometric.datasets.md17 as md17_module
md17_module.MD17.revised_url = (
    'https://figshare.com/ndownloader/files/30631036'
)
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, RadiusGraph
from torch_geometric.data import Data
from torch_geometric.nn import SchNet


SEED = 42

# ── QM9 target metadata ────────────────────────────────────────────────────────
# index : (name, unit)
QM9_TARGETS = {
    0:  ('mu',    'D'),        # dipole moment
    1:  ('alpha', 'a_0^3'),    # isotropic polarizability
    2:  ('HOMO',  'eV'),       # highest occupied molecular orbital energy
    3:  ('LUMO',  'eV'),       # lowest unoccupied molecular orbital energy
    4:  ('gap',   'eV'),       # HOMO-LUMO gap
    5:  ('R2',    'a_0^2'),    # electronic spatial extent
    6:  ('ZPVE',  'eV'),       # zero point vibrational energy
    7:  ('U0',    'eV'),       # internal energy at 0 K
    8:  ('U',     'eV'),       # internal energy at 298.15 K
    9:  ('H',     'eV'),       # enthalpy at 298.15 K
    10: ('G',     'eV'),       # free energy at 298.15 K
    11: ('Cv',    'cal/mol·K') # heat capacity at 298.15 K
}

# ── MD17 available molecules ────────────────────────────────────────────────────
RMD17_MOLECULES = [
    'benzene', 'uracil', 'naphthalene', 'aspirin',
    'salicylic acid', 'malonaldehyde', 'ethanol', 'toluene',
]

def set_seed(seed=SEED):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    return device


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Dataset loading ─────────────────────────────────────────────────────────────

def get_qm9_data(target, batch_size=256, root='data/QM9', seed=SEED, train_size=100000, val_size=10000, test_size=10000):
    """Load QM9 and return train/val/test loaders with normalization stats.

    Standard split: 100k / 10k / rest (~28k).
    Returns loaders and (mean, std) of the target over the training set,
    which the caller can use for normalization if desired.
    """
    transform = RadiusGraph(r=5.0, max_num_neighbors=32)

    dataset = QM9(root=root, transform=transform)
    target_name, target_unit = QM9_TARGETS[target]
    print(f'QM9 loaded: {len(dataset)} molecules')
    print(f'Target {target}: {target_name} [{target_unit}]')

    torch.manual_seed(seed)
    perm = torch.randperm(len(dataset))
    dataset = dataset[perm]

    train_dataset = dataset[:train_size]
    val_dataset   = dataset[train_size:train_size + val_size]
    test_dataset  = dataset[train_size + val_size:train_size + val_size+ test_size]

    # compute normalization stats on training set
    train_targets = torch.stack([d.y[:, target] for d in train_dataset])
    mean = train_targets.mean().item()
    std  = train_targets.std().item()

    print(f'Split: {len(train_dataset)} train | {len(val_dataset)} val | {len(test_dataset)} test')
    print(f'Target stats (train): mean={mean:.4f}, std={std:.4f}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size)

    return train_loader, val_loader, test_loader, (mean, std)


# mapping from clean names to npz filenames
RMD17_FILE_MAP = {
    'aspirin':        'rmd17_aspirin.npz',
    'benzene':        'rmd17_benzene.npz',
    'ethanol':        'rmd17_ethanol.npz',
    'malonaldehyde':  'rmd17_malonaldehyde.npz',
    'naphthalene':    'rmd17_naphthalene.npz',
    'salicylic acid': 'rmd17_salicylic.npz',
    'toluene':        'rmd17_toluene.npz',
    'uracil':         'rmd17_uracil.npz',
    'azobenzene':     'rmd17_azobenzene.npz',
    'paracetamol':    'rmd17_paracetamol.npz',
}

def get_md17_data(molecule='aspirin', batch_size=64,
                  split_idx=1,
                  root='data/MD17/rmd17',
                  cutoff=5.0, seed=SEED):
    """Load revised MD17 using official splits.
    
    Uses rMD17 official temporally-decorrelated splits (1-5).
    Training: 1000 configurations from split. Validation: held out from train.
    Test: all remaining configurations from the full trajectory.
    """
    filename = RMD17_FILE_MAP.get(molecule)
    if filename is None:
        raise ValueError(f'Unknown molecule: {molecule}')

    path = os.path.join(root, filename)
    raw = np.load(path)

    coords   = torch.tensor(raw['coords'], dtype=torch.float32)
    energies = torch.tensor(raw['energies'], dtype=torch.float32)
    forces   = torch.tensor(raw['forces'], dtype=torch.float32)
    z        = torch.tensor(raw['nuclear_charges'], dtype=torch.long)

    n_conf = coords.shape[0]
    max_z  = int(z.max().item()) + 1
    x_onehot = nn.functional.one_hot(z, num_classes=max_z).float()

    # Load official split indices
    splits_dir = os.path.join(root, 'splits')
    train_idx = np.loadtxt(
        os.path.join(splits_dir, f'index_train_{split_idx:02d}.csv'),
        dtype=int,
    )
    test_idx = np.loadtxt(
        os.path.join(splits_dir, f'index_test_{split_idx:02d}.csv'),
        dtype=int,
    )

    val_size = 100
    val_idx = train_idx[-val_size:]
    train_idx = train_idx[:-val_size]

    print(f'rMD17 loaded: {molecule} ({n_conf} conformations, {z.shape[0]} atoms)')
    print(f'Using official split {split_idx}')
    print(f'Split: {len(train_idx)} train | {len(val_idx)} val | {len(test_idx)} test')

    # Build Data objects for the indices we actually use
    rg = RadiusGraph(r=cutoff, loop=False)
    
    def build(idx_array):
        out = []
        for i in idx_array:
            d = Data(
                x=x_onehot,
                pos=coords[i],
                energy=energies[i].unsqueeze(0),
                force=forces[i],
                z=z,
            )
            d = rg(d)
            out.append(d)
        return out
    
    train_dataset = build(train_idx)
    val_dataset   = build(val_idx)
    test_dataset  = build(test_idx)

    train_e = torch.stack([d.energy for d in train_dataset])
    mean = train_e.mean().item()
    std  = train_e.std().item()

    print(f'Energy stats (train): mean={mean:.6f}, std={std:.6f}')
    print(f'Node features: {x_onehot.shape[1]}-dim one-hot  |  Cutoff: {cutoff} A')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size)

    return train_loader, val_loader, test_loader, (mean, std), x_onehot.shape[1]


# ── Model building ──────────────────────────────────────────────────────────────

class EGNN_QM9_Wrapper(nn.Module):
    """Wraps the raw EGNN to accept PyG Data objects and return graph-level predictions."""

    def __init__(self, egnn):
        super().__init__()
        self.egnn = egnn
        from torch_geometric.nn import global_mean_pool
        self._pool = global_mean_pool

    def forward(self, data):
        h = data.x                        # (num_nodes, 11) node features in QM9
        x = data.pos                      # 3D coordinates
        edges = data.edge_index
        edge_attr = data.edge_attr        # (num_edges, 4) bond features in QM9

        h_out, _ = self.egnn(h, x, edges, edge_attr)

        # pool node features to graph-level
        out = self._pool(h_out, data.batch)
        return out



class SchNetWrapper(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.schnet = SchNet(**kwargs)

    def forward(self, data):
        return self.schnet(data.z, data.pos, data.batch)

def build_egnn(in_dim, hid_dim=128, out_dim=1, n_layers=4):
    """Build an EGNN model wrapped for PyG Data input (QM9)."""
    from models.EGNN import EGNN
    egnn = EGNN(
        in_node_nf=in_dim, hidden_nf=hid_dim, n_layers=n_layers,
        out_node_nf=out_dim, in_edge_nf=0
    )
    model = EGNN_QM9_Wrapper(egnn)
    print(f'EGNN: {count_parameters(model):,} parameters')
    return model


def build_cgenn(in_dim, hid_dim=64, out_dim=1, n_layers=4):
    """Build an EGNN model wrapped for PyG Data input (QM9)."""
    from models.CGENN import CGGNN
    cgenn = CGGNN(in_features=in_dim+1, hidden_features=hid_dim, out_features=out_dim, n_layers=n_layers)
    model = cgenn
    print(f'CGENN: {count_parameters(model):,} parameters')
    return model

def build_gat(in_dim=14, hid_dim=64, out_dim=1, n_layers=4, n_heads=4):
    from models.GAT import GAT
    model = GAT(node_input_dim=in_dim, hidden_dim=hid_dim, out_dim=out_dim, num_layers=n_layers, num_heads=n_heads, num_rbf=20, cutoff=10)
    print(f'GAT: {count_parameters(model):,} parameters')
    return model


def build_ga_gat(in_dim=14, hid_dim=32, out_dim=1, n_layers=4, n_heads=4):
    from models.GAGAT import GA_GAT
    model = GA_GAT(hidden_dim=hid_dim, out_dim=out_dim, num_layers=n_layers, num_heads=n_heads, num_rbf=20, cutoff=10)
    print(f'GAT: {count_parameters(model):,} parameters')
    return model

# ── Checkpointing ───────────────────────────────────────────────────────────────

def save_checkpoint(state, save_dir='pretrained_models'):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, state['name'] + '.pt')
    torch.save(state, path)
    print(f'Checkpoint saved: {path}')


def load_checkpoint(path, device='cpu'):
    state = torch.load(path, map_location=device, weights_only=False)
    print(f'Loaded checkpoint: {path} (epoch {state.get("epoch", "?")})')
    return state


# ── Evaluation helpers ──────────────────────────────────────────────────────────

def evaluate_mae(model, loader, target_idx, device, reduction='mean'):
    """Compute MAE over an entire loader for a QM9 target."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data).squeeze()
            labels = data.y[:, target_idx]
            if reduction == 'sum':
                total_loss += nn.functional.l1_loss(out, labels, reduction='sum').item()
            else:
                total_loss += nn.functional.l1_loss(out, labels).item() * data.num_graphs
            total_samples += data.num_graphs

    return total_loss / total_samples


def evaluate_mae_energy(model, loader, mean, std, device):
    """Compute MAE over an entire loader for MD17 energy prediction."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data).squeeze()
            out = out*std + mean
            labels = data.energy.squeeze()
            total_loss += nn.functional.l1_loss(out, labels, reduction='sum').item()
            total_samples += data.num_graphs

    return total_loss / total_samples


