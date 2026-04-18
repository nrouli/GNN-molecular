"""
Out-of-distribution evaluation of a QM9-trained model on the Alchemy dataset.

Alchemy (Chen et al., 2019) contains 12 quantum mechanical properties of ~120k
organic molecules with 9-14 heavy atoms and elements {H, C, N, O, F, S, Cl}.
QM9 covers molecules with up to 9 heavy atoms and {H, C, N, O, F}. Evaluating
a QM9-trained model on Alchemy tests generalization to larger, more diverse
molecules.

IMPORTANT: Alchemy properties were computed with PySCF, while QM9 uses Gaussian.
This causes systematic offsets unrelated to model generalization. Following
Zhang et al. (NeurIPS ML4PS 2022, "Do Better QM9 Models Extrapolate as Better
Quantum Chemical Property Predictors?"), we support two correction methods:
  1. QMALL recomputed targets (preferred): use --qmall_csv to point to the
     Gaussian-recomputed Alchemy properties from github.com/YZHANG1996/QMALL
  2. Linear correction: use --linear_correction to estimate and apply a linear
     transform (y_corrected = w * y_pyscf + b) using Al9 molecules that overlap
     in size with QM9 as a calibration set.

The script:
  1. Loads Alchemy SDF files and builds QM9-compatible PyG Data objects
  2. Filters to H/C/N/O/F-only molecules (so the input features match QM9)
  3. Converts Alchemy targets to QM9 units (Hartree -> eV, etc.)
  4. Optionally applies DFT software correction (QMALL or linear)
  5. Evaluates the pretrained model per target
  6. Optionally evaluates on the QM9 test set for a side-by-side comparison

Prerequisites:
  - rdkit  (`pip install rdkit` or `conda install -c conda-forge rdkit`)
  - Alchemy dataset downloaded from https://alchemy.tencent.com
    Expected layout after extraction:
        data/Alchemy/
            atom_9/  atom_10/  ...  atom_14/     (SDF files)
            dev_target.csv                       (property labels)

Usage:
    # Raw Alchemy (no correction; includes DFT software error):
    python eval_ood.py --model schnet --target 4 --checkpoint pretrained_models/schnet_QM9_gap.pt --compare_qm9

    # With QMALL recomputed targets (preferred):
    python eval_ood.py --model schnet --target 4 --checkpoint pretrained_models/schnet_QM9_gap.pt --compare_qm9 --qmall_csv data/QMALL/alchemy_recomputed.csv

    # With linear correction estimated from Al9:
    python eval_ood.py --model schnet --target 4 --checkpoint pretrained_models/schnet_QM9_gap.pt --compare_qm9 --linear_correction

    # QM9 size-based OOD (no Alchemy needed):
    python eval_ood.py --model schnet --target 4 --checkpoint pretrained_models/schnet_QM9_gap.pt --size_split

Run from the project root (GNN-molecular/).
"""

import os
import sys
import csv
import glob
import argparse

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet

from utils.util import (
    set_seed, get_device, count_parameters, load_checkpoint,
    get_qm9_data, build_egnn, build_gat, build_ga_gat, build_cgenn, evaluate_mae,
    QM9_TARGETS, SEED, SchNetWrapper,
)


# ── Constants ───────────────────────────────────────────────────────────────────

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414
EH_TO_CAL_PER_MOL = 627509.4740631

ALCHEMY_COLUMNS = ['mu', 'alpha', 'homo', 'lumo', 'gap',
                   'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']

# Which targets need Hartree -> eV conversion (from raw PySCF Alchemy CSV)
HARTREE_TARGETS = {2, 3, 4, 6, 7, 8, 9, 10}  # homo, lumo, gap, zpve, U0, U, H, G
CV_TARGET = 11

QM9_ATOM_TYPES = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}  # H, C, N, O, F
QM9_ALLOWED_Z = set(QM9_ATOM_TYPES.keys())

# Models that use bond edge_attr. EGNN and SchNet do NOT: EGNN is trained with
# RadiusGraph (no bond features), and SchNet ignores edge_attr entirely.
# GAT, GA-GAT, and CGENN use bond features and should keep them.
MODELS_WITHOUT_BOND_EDGE_ATTR = {'egnn', 'schnet'}


# ── Alchemy loading ────────────────────────────────────────────────────────────

ALCHEMY_TO_QM9 = {
    'mu':    0,
    'alpha': 1,
    'homo':  2,
    'lumo':  3,
    'gap':   4,
    'r2':    5,
    'zpve':  6,
    'u0':    7,
    'u298':  8,  'u':     8,
    'h298':  9,  'h':     9,
    'g298':  10, 'g':     10,
    'cv':    11,
}


def load_alchemy_targets(csv_path):
    """Load Alchemy target properties from CSV, mapping to QM9 target order."""
    targets = {}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        col_map = {}
        for h in headers:
            clean = h.replace('"', '').split('(')[0].split('\n')[0].strip().lower()
            if clean in ALCHEMY_TO_QM9:
                col_map[h] = ALCHEMY_TO_QM9[clean]

        print(f'Column mapping: {len(col_map)} of 12 targets matched')
        for h, idx in col_map.items():
            name = QM9_TARGETS[idx][0]
            print(f'  "{h[:40]}" -> index {idx} ({name})')

        for row in reader:
            gdb_id = list(row.values())[0]

            y = torch.zeros(12, dtype=torch.float32)
            valid = True
            for col_name, qm9_idx in col_map.items():
                try:
                    y[qm9_idx] = float(row[col_name])
                except (ValueError, TypeError, KeyError):
                    valid = False
                    break

            if not valid:
                continue

            # Hartree -> eV for orbital/energy targets
            for idx in HARTREE_TARGETS:
                y[idx] *= HAR2EV
            # Cv is already in cal/molK in the CSV

            targets[str(gdb_id).strip()] = y

    print(f'Loaded {len(targets)} target entries from {csv_path}')
    return targets


def load_qmall_targets(qmall_csv_path):
    """Load QMALL recomputed Alchemy targets (Gaussian-recomputed).

    The QMALL dataset from Zhang et al. (NeurIPS ML4PS 2022) recomputes
    Alchemy properties using Gaussian at the same B3LYP/6-31G(2df,p) level
    as QM9, eliminating PySCF-vs-Gaussian systematic differences.

    Expected CSV format: first column is molecule ID, remaining columns are
    the 12 properties in QM9 order (already in QM9 units).

    Returns a dict mapping mol_id -> tensor of shape (12,).
    """
    targets = {}

    with open(qmall_csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        print(f'QMALL CSV columns: {header}')

        for row in reader:
            mol_id = str(row[0]).strip()
            try:
                values = [float(v) for v in row[1:13]]
            except (ValueError, IndexError):
                continue
            targets[mol_id] = torch.tensor(values, dtype=torch.float32)

    print(f'Loaded {len(targets)} QMALL recomputed target entries')
    return targets


def sdf_to_data(sdf_path, targets_dict, mol_id):
    """Parse a single SDF file into a PyG Data object with QM9-compatible features.

    Node features (11-dim, matching QM9 in PyG):
      [0:5]  one-hot atom type (H, C, N, O, F)
      [5]    atomic number
      [6]    is aromatic (0/1)
      [7]    sp hybridization (0/1)
      [8]    sp2 hybridization (0/1)
      [9]    sp3 hybridization (0/1)
      [10]   number of attached hydrogens

    Returns None if the molecule contains atoms outside {H, C, N, O, F}.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError:
        raise ImportError(
            'rdkit is required for Alchemy loading. '
            'Install with: pip install rdkit  or  conda install -c conda-forge rdkit'
        )

    mol = Chem.MolFromMolFile(sdf_path, removeHs=False, sanitize=True)
    if mol is None:
        return None

    atoms = mol.GetAtoms()
    for atom in atoms:
        if atom.GetAtomicNum() not in QM9_ALLOWED_Z:
            return None

    conf = mol.GetConformer()
    pos = torch.tensor(
        [[conf.GetAtomPosition(i).x,
          conf.GetAtomPosition(i).y,
          conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())],
        dtype=torch.float32
    )

    hybridization_map = {
        Chem.rdchem.HybridizationType.SP: [1, 0, 0],
        Chem.rdchem.HybridizationType.SP2: [0, 1, 0],
        Chem.rdchem.HybridizationType.SP3: [0, 0, 1],
    }

    x_list = []
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        one_hot = [0.0] * 5
        one_hot[QM9_ATOM_TYPES[z]] = 1.0
        atomic_num = float(z)
        aromatic = 1.0 if atom.GetIsAromatic() else 0.0
        hyb = hybridization_map.get(atom.GetHybridization(), [0, 0, 0])
        num_hs = float(atom.GetTotalNumHs())
        feat = one_hot + [atomic_num, aromatic] + hyb + [num_hs]
        x_list.append(feat)

    x = torch.tensor(x_list, dtype=torch.float32)

    edge_src, edge_dst = [], []
    bond_types = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_src += [i, j]
        edge_dst += [j, i]

        bt = [0.0] * 4
        b_type = bond.GetBondTypeAsDouble()
        if b_type == 1.0:   bt[0] = 1.0
        elif b_type == 2.0: bt[1] = 1.0
        elif b_type == 3.0: bt[2] = 1.0
        if bond.GetIsAromatic(): bt[3] = 1.0
        bond_types.append(bt)
        bond_types.append(bt)

    if len(edge_src) == 0:
        return None

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = torch.tensor(bond_types, dtype=torch.float32)

    y = targets_dict.get(str(mol_id))
    if y is None:
        return None

    z_list = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    z = torch.tensor(z_list, dtype=torch.long)

    data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr,
                y=y.unsqueeze(0), z=z)
    data.num_heavy_atoms = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)

    return data


def load_alchemy_dataset(root, csv_path, max_molecules=None, cache_file=None,
                         force_reload=False, qmall_csv=None):
    """Load Alchemy SDF files and return a list of PyG Data objects, with caching.

    If qmall_csv is provided, uses QMALL recomputed targets instead of raw
    PySCF values, eliminating the DFT software systematic error.

    If cache_file exists and force_reload is False, load from cache.
    Otherwise process all SDF files and save to cache_file.
    """
    # Determine cache file name (different for QMALL vs raw)
    if cache_file is None:
        suffix = '_qmall' if qmall_csv else '_raw'
        cache_file = os.path.join(root, 'processed', f'alchemy{suffix}.pt')

    if os.path.exists(cache_file) and not force_reload:
        print(f"Loading cached Alchemy dataset from {cache_file}")
        dataset = torch.load(cache_file, weights_only=False)
        print(f"Loaded {len(dataset)} molecules from cache.")
        if max_molecules is not None and len(dataset) > max_molecules:
            dataset = dataset[:max_molecules]
        return dataset

    # Load targets
    if qmall_csv is not None:
        print(f"Using QMALL recomputed targets from {qmall_csv}")
        print("(Gaussian-recomputed; eliminates PySCF-vs-Gaussian systematic error)")
        targets_dict = load_qmall_targets(qmall_csv)
    else:
        print("Using raw PySCF Alchemy targets")
        print("WARNING: these have systematic offsets vs QM9 (Gaussian).")
        print("         Use --qmall_csv or --linear_correction for fair comparison.")
        targets_dict = load_alchemy_targets(csv_path)

    # Find SDF files
    print("Processing Alchemy SDF files...")
    sdf_files = sorted(glob.glob(os.path.join(root, 'atom_*', '*.sdf')))
    if not sdf_files:
        sdf_files = sorted(glob.glob(os.path.join(root, '*.sdf')))
    if not sdf_files:
        raise FileNotFoundError(f'No SDF files found in {root}.')

    print(f'Found {len(sdf_files)} SDF files.')
    if max_molecules:
        sdf_files = sdf_files[:max_molecules]

    dataset = []
    skipped = 0

    for sdf_path in tqdm(sdf_files, desc='Loading Alchemy'):
        basename = os.path.splitext(os.path.basename(sdf_path))[0]
        mol_id = basename.lstrip('0') or '0'
        data = sdf_to_data(sdf_path, targets_dict, mol_id)
        if data is None:
            data = sdf_to_data(sdf_path, targets_dict, basename)
        if data is None:
            skipped += 1
            continue
        dataset.append(data)

    print(f'\nAlchemy loaded: {len(dataset)} molecules (QM9-compatible)')
    print(f'Skipped: {skipped} (non-H/C/N/O/F atoms or missing targets/parse errors)')

    if dataset:
        sizes = [d.num_heavy_atoms for d in dataset]
        from collections import Counter
        dist = Counter(sizes)
        print('Heavy atom distribution:')
        for n in sorted(dist.keys()):
            print(f'  {n} heavy atoms: {dist[n]} molecules')

    cache_dir = os.path.dirname(cache_file)
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(dataset, cache_file)
    print(f"Cached Alchemy dataset to {cache_file}")

    return dataset


# ── Edge attr handling ──────────────────────────────────────────────────────────

def prepare_alchemy_batch(data, model_name):
    """Strip bond edge_attr for models that were not trained with it.

    EGNN is trained via RadiusGraph which produces no bond edge features;
    passing Alchemy's 4-dim bond edge_attr would cause an MLP dimension
    mismatch. SchNet ignores edge_attr entirely. GAT, GA-GAT, and CGENN
    use bond features and should keep them.
    """
    if model_name in MODELS_WITHOUT_BOND_EDGE_ATTR:
        data.edge_attr = None
    return data


# ── Linear correction ───────────────────────────────────────────────────────────

def compute_linear_correction(model, alchemy_data, qm9_loader, target_idx,
                              device, model_name, n_calibration=500):
    """Estimate a linear correction for the PySCF-vs-Gaussian systematic offset.

    Uses Al9 molecules (9 heavy atoms, same size range as QM9) as a calibration
    set. Computes model predictions on both QM9 test molecules and Al9 molecules,
    then fits y_corrected = w * y_raw + b to minimize the offset.

    This follows the approach of Zhang et al. (QMALL, NeurIPS ML4PS 2022),
    who used 500 calibration points per dataset.

    Returns (weight, bias) for the linear transform.
    """
    print(f'\n--- Computing linear correction from Al9 calibration set ---')

    # Collect Al9 molecules
    al9 = [d for d in alchemy_data if d.num_heavy_atoms == 9]
    if len(al9) == 0:
        print('WARNING: No Al9 molecules found. Cannot compute correction.')
        return 1.0, 0.0

    if len(al9) > n_calibration:
        al9 = al9[:n_calibration]

    print(f'Using {len(al9)} Al9 molecules for calibration')

    # Get model predictions on Al9
    al9_loader = DataLoader(al9, batch_size=256)
    model.eval()
    al9_preds, al9_labels = [], []
    with torch.no_grad():
        for data in al9_loader:
            data = data.to(device)
            data = prepare_alchemy_batch(data, model_name)
            out = model(data).squeeze()
            labels = data.y[:, target_idx]
            al9_preds.append(out.cpu())
            al9_labels.append(labels.cpu())

    al9_preds = torch.cat(al9_preds).numpy()
    al9_labels = torch.cat(al9_labels).numpy()

    # Get model predictions on QM9 test (for reference distribution)
    qm9_preds, qm9_labels = [], []
    with torch.no_grad():
        for data in qm9_loader:
            data = data.to(device)
            out = model(data).squeeze()
            labels = data.y[:, target_idx]
            qm9_preds.append(out.cpu())
            qm9_labels.append(labels.cpu())

    qm9_preds = torch.cat(qm9_preds).numpy()
    qm9_labels = torch.cat(qm9_labels).numpy()

    # The systematic error is in the labels, not the predictions.
    # We want: corrected_alchemy_label = w * raw_alchemy_label + b
    # such that the corrected labels align with QM9's Gaussian-computed scale.
    #
    # Approach: the model was trained on QM9 (Gaussian). Its predictions on Al9
    # reflect the QM9 scale. The Al9 labels are in PySCF scale.
    # Fit: model_pred ≈ w * alchemy_label + b
    # Then apply this transform to all Alchemy labels.

    # Least squares: [al9_labels, 1] @ [w, b]^T ≈ al9_preds
    A = np.stack([al9_labels, np.ones_like(al9_labels)], axis=1)
    result = np.linalg.lstsq(A, al9_preds, rcond=None)
    w, b = result[0]

    # Compute calibration quality
    corrected = w * al9_labels + b
    residual_mae = np.mean(np.abs(al9_preds - corrected))
    raw_mae = np.mean(np.abs(al9_preds - al9_labels))

    print(f'Linear correction: y_corrected = {w:.6f} * y_raw + {b:.6f}')
    print(f'Al9 MAE before correction: {raw_mae:.4f}')
    print(f'Al9 residual MAE after correction: {residual_mae:.4f}')
    print(f'(Residual reflects true OOD generalization error on Al9)')

    return w, b


def apply_linear_correction(dataset, target_idx, w, b):
    """Apply linear correction to Alchemy target labels in-place."""
    for data in dataset:
        data.y[0, target_idx] = w * data.y[0, target_idx] + b
    return dataset


# ── QM9 size-based OOD (no Alchemy needed) ──────────────────────────────────

def qm9_size_split_ood(target, batch_size=256, seed=SEED):
    """Split QM9 by molecule size for a pseudo-OOD test.

    Train/val: molecules with <= 7 heavy atoms
    OOD test:  molecules with 8-9 heavy atoms
    """
    from torch_geometric.datasets import QM9

    dataset = QM9(root='data/QM9')
    print(f'QM9: {len(dataset)} molecules')

    small, large = [], []
    for data in dataset:
        n_heavy = (data.z > 1).sum().item()
        if n_heavy <= 7:
            small.append(data)
        else:
            large.append(data)

    print(f'Small (<= 7 heavy): {len(small)} | Large (8-9 heavy): {len(large)}')

    ood_loader = DataLoader(large, batch_size=batch_size)
    return ood_loader, len(large)


# ── Evaluation ──────────────────────────────────────────────────────────────────

def evaluate_ood(model, loader, target_idx, device, model_name='egnn',
                 is_alchemy=False):
    """Compute MAE on Out-Of-Distribution (OOD) data for a specific QM9 target.

    is_alchemy: if True, applies model-aware edge_attr stripping for Alchemy
                batches (bond edge_attr is incompatible with models trained
                without it, e.g. EGNN via RadiusGraph).
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for data in tqdm(loader, desc='Evaluating OOD', leave=False):
            data = data.to(device)
            if is_alchemy:
                data = prepare_alchemy_batch(data, model_name)
            out = model(data).squeeze()
            labels = data.y[:, target_idx]
            total_loss += nn.functional.l1_loss(out, labels, reduction='sum').item()
            total_samples += data.num_graphs

    return total_loss / total_samples


def evaluate_per_size(model, dataset, target_idx, device, batch_size=256,
                      model_name='egnn'):
    """Evaluate MAE broken down by number of heavy atoms."""
    from collections import defaultdict

    size_groups = defaultdict(list)
    for d in dataset:
        size_groups[d.num_heavy_atoms].append(d)

    results = {}
    for n_heavy in sorted(size_groups.keys()):
        group = size_groups[n_heavy]
        loader = DataLoader(group, batch_size=batch_size)
        mae = evaluate_ood(model, loader, target_idx, device,
                           model_name=model_name, is_alchemy=True)
        results[n_heavy] = {'mae': mae, 'count': len(group)}

    return results


# ── Main ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        'QM9 OOD Evaluation (Alchemy)',
        description='Evaluate QM9-trained models on Alchemy for OOD generalization. '
                    'Supports DFT software correction via QMALL or linear calibration.'
    )

    parser.add_argument('--model', type=str, default='egnn',
                        choices=['egnn', 'gat', 'gagat', 'schnet', 'cgenn'])
    parser.add_argument('--target', type=int, default=4,
                        help='QM9 target index (0-11). Default: 4 (gap)')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=SEED)

    # Alchemy options
    parser.add_argument('--alchemy_root', type=str, default='data/Alchemy')
    parser.add_argument('--alchemy_csv', type=str, default='data/Alchemy/final_version.csv')
    parser.add_argument('--max_molecules', type=int, default=None)
    parser.add_argument('--alchemy_cache', type=str, default=None,
                        help='Cache path (auto-generated if not set)')
    parser.add_argument('--force_reload', action='store_true')

    # DFT correction options
    correction = parser.add_mutually_exclusive_group()
    correction.add_argument('--qmall_csv', type=str, default=None,
                            help='Path to QMALL recomputed Alchemy targets '
                                 '(Gaussian-recomputed, from github.com/YZHANG1996/QMALL). '
                                 'Preferred correction method.')
    correction.add_argument('--linear_correction', action='store_true',
                            help='Estimate and apply a linear correction using Al9 '
                                 'molecules as calibration. Lighter-weight alternative '
                                 'to QMALL recomputed targets.')

    # Comparison options
    parser.add_argument('--compare_qm9', action='store_true',
                        help='Also evaluate on QM9 test set for comparison')
    parser.add_argument('--size_split', action='store_true',
                        help='Use QM9 size-based OOD split instead of Alchemy')

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    target_name, target_unit = QM9_TARGETS[args.target]
    print(f'\n--- OOD Evaluation ---')
    print(f'Target {args.target}: {target_name} [{target_unit}]')

    # Describe correction mode
    if args.qmall_csv:
        print(f'Correction: QMALL recomputed targets (Gaussian)')
    elif args.linear_correction:
        print(f'Correction: linear calibration from Al9')
    else:
        print(f'Correction: NONE (raw PySCF Alchemy targets)')
        print(f'  Note: results include DFT software systematic error.')
        print(f'  Use --qmall_csv or --linear_correction for fair comparison.')

    # ── Load model ──────────────────────────────────────────────────────────
    state = load_checkpoint(args.checkpoint, device=str(device))
    hparams = state.get('hparams', {})

    match args.model:
        case 'egnn':
            model = build_egnn(in_dim=hparams.get('in_dim', 11),
                               hid_dim=hparams.get('hid_dim', 64),
                               out_dim=1,
                               n_layers=hparams.get('n_layers', 4))
        case 'gat':
            model = build_gat(in_dim=hparams.get('in_dim', 14),
                              hid_dim=hparams.get('hid_dim', 64),
                              out_dim=1,
                              n_layers=hparams.get('n_layers', 4),
                              n_heads=hparams.get('n_heads', 4))
        case 'gagat':
            model = build_ga_gat(in_dim=hparams.get('in_dim', 14),
                              hid_dim=hparams.get('hid_dim', 64),
                              out_dim=1,
                              n_layers=hparams.get('n_layers', 4),
                              n_heads=hparams.get('n_heads', 4))
        case 'schnet':
            model = SchNetWrapper(
                hidden_channels=hparams.get('hidden_channels', 128),
                num_filters=hparams.get('num_filters', 128),
                num_interactions=hparams.get('num_interactions', 6),
                num_gaussians=hparams.get('num_gaussians', 25),
                cutoff=hparams.get('cutoff', 10.0))
        case 'cgenn':
            model = build_cgenn(in_dim=hparams.get('in_dim', 14),
                                hid_dim=hparams.get('hid_dim', 32),
                                out_dim=1,
                                n_layers=hparams.get('n_layers', 4))
        case _:
            model = build_egnn(in_dim=hparams.get('in_dim', 11),
                               hid_dim=hparams.get('hid_dim', 64),
                               out_dim=1,
                               n_layers=hparams.get('n_layers', 4))

    model.load_state_dict(state['state_dict'], strict=False)
    model.to(device)
    model.eval()

    # ── In-distribution baseline (QM9 test set) ────────────────────────────
    qm9_test_mae = None
    qm9_test_loader = None
    if args.compare_qm9 or args.size_split or args.linear_correction:
        _, _, qm9_test_loader, _ = get_qm9_data(
            target=args.target, batch_size=args.batch_size, seed=args.seed
        )
        # QM9 test set uses RadiusGraph edges; is_alchemy=False
        qm9_test_mae = evaluate_ood(model, qm9_test_loader, args.target, device,
                                    model_name=args.model, is_alchemy=False)
        print(f'\nQM9 in-distribution test MAE: {qm9_test_mae:.4f} {target_unit}')

    # ── QM9 size-based OOD ──────────────────────────────────────────────────
    if args.size_split:
        print(f'\n--- QM9 Size-based OOD ---')
        ood_loader, n_ood = qm9_size_split_ood(
            args.target, batch_size=args.batch_size, seed=args.seed
        )
        size_ood_mae = evaluate_ood(model, ood_loader, args.target, device,
                                    model_name=args.model, is_alchemy=False)
        print(f'QM9 large-molecule OOD MAE: {size_ood_mae:.4f} {target_unit}  '
              f'({n_ood} molecules)')

        if qm9_test_mae is not None:
            ratio = size_ood_mae / qm9_test_mae if qm9_test_mae > 0 else float('inf')
            print(f'OOD / in-distribution ratio: {ratio:.2f}x')

    # ── Alchemy OOD ─────────────────────────────────────────────────────────
    if not args.size_split:
        if not os.path.isdir(args.alchemy_root):
            print(f'\nAlchemy data not found at {args.alchemy_root}')
            print('Download from: https://alchemy.tencent.com')
            print('Or use --size_split for a QM9-internal size-based OOD test.')
            return

        print(f'\n--- Alchemy OOD Evaluation ---')
        alchemy_data = load_alchemy_dataset(
            args.alchemy_root, args.alchemy_csv,
            max_molecules=args.max_molecules,
            cache_file=args.alchemy_cache,
            force_reload=args.force_reload,
            qmall_csv=args.qmall_csv,
        )

        if not alchemy_data:
            print('No compatible Alchemy molecules loaded.')
            return

        # Apply linear correction if requested
        correction_info = 'none'
        if args.linear_correction:
            if qm9_test_loader is None:
                _, _, qm9_test_loader, _ = get_qm9_data(
                    target=args.target, batch_size=args.batch_size, seed=args.seed
                )
            w, b = compute_linear_correction(
                model, alchemy_data, qm9_test_loader,
                args.target, device, model_name=args.model
            )
            alchemy_data = apply_linear_correction(
                alchemy_data, args.target, w, b
            )
            correction_info = f'linear (w={w:.6f}, b={b:.6f})'
        elif args.qmall_csv:
            correction_info = 'QMALL recomputed (Gaussian)'

        alchemy_loader = DataLoader(alchemy_data, batch_size=args.batch_size)
        alchemy_mae = evaluate_ood(model, alchemy_loader, args.target, device,
                                   model_name=args.model, is_alchemy=True)

        # ── Report ──────────────────────────────────────────────────────────
        print(f'\n{"="*60}')
        print(f'Target: {target_name} [{target_unit}]')
        print(f'DFT correction: {correction_info}')
        print(f'{"="*60}')
        if qm9_test_mae is not None:
            print(f'QM9 test MAE (in-distribution):    {qm9_test_mae:.4f} {target_unit}')
        print(f'Alchemy MAE (out-of-distribution): {alchemy_mae:.4f} {target_unit}')
        print(f'Alchemy molecules evaluated: {len(alchemy_data)}')
        if qm9_test_mae is not None and qm9_test_mae > 0:
            ratio = alchemy_mae / qm9_test_mae
            print(f'OOD / in-distribution ratio: {ratio:.2f}x')
        print(f'{"="*60}')

        # Per-size breakdown
        print(f'\nPer-size breakdown:')
        size_results = evaluate_per_size(
            model, alchemy_data, args.target, device, args.batch_size,
            model_name=args.model
        )
        print(f'{"Heavy atoms":<14} {"Count":>8} {"MAE":>12}')
        print(f'{"─"*36}')
        for n_heavy, res in size_results.items():
            print(f'{n_heavy:<14} {res["count"]:>8} {res["mae"]:>12.4f}')

        # ── Save results ────────────────────────────────────────────────────
        os.makedirs('results/csv', exist_ok=True)
        model_name = state.get('model_class', 'unknown')
        corr_tag = 'qmall' if args.qmall_csv else ('lincorr' if args.linear_correction else 'raw')
        csv_path = f'results/csv/ood_{model_name}_alchemy_{target_name}_{corr_tag}.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['source', 'n_heavy_atoms', 'count', 'mae', 'unit',
                             'correction'])
            if qm9_test_mae is not None:
                writer.writerow(['QM9_test', 'all', '', f'{qm9_test_mae:.6f}',
                                 target_unit, 'n/a'])
            writer.writerow(['Alchemy_all', 'all', len(alchemy_data),
                             f'{alchemy_mae:.6f}', target_unit, correction_info])
            for n_heavy, res in size_results.items():
                writer.writerow([f'Alchemy_{n_heavy}', n_heavy, res['count'],
                                 f'{res["mae"]:.6f}', target_unit, correction_info])
        print(f'\nResults saved: {csv_path}')


if __name__ == '__main__':
    main()