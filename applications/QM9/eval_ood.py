"""
Out-of-distribution evaluation of a QM9-trained model on the Alchemy dataset.

Alchemy (Chen et al., 2019) contains 12 quantum mechanical properties of ~120k
organic molecules with 9-14 heavy atoms and elements {H, C, N, O, F, S, Cl}.
QM9 covers molecules with up to 9 heavy atoms and {H, C, N, O, F}. Evaluating
a QM9-trained model on Alchemy tests generalization to larger, more diverse
molecules.

The script:
  1. Loads Alchemy SDF files and builds QM9-compatible PyG Data objects
  2. Filters to H/C/N/O/F-only molecules (so the input features match QM9)
  3. Converts Alchemy targets to QM9 units (Hartree -> eV, etc.)
  4. Evaluates the pretrained model per target
  5. Optionally evaluates on the QM9 test set for a side-by-side comparison

Prerequisites:
  - rdkit  (`pip install rdkit` or `conda install -c conda-forge rdkit`)
  - Alchemy dataset downloaded from https://alchemy.tencent.com
    Expected layout after extraction:
        data/Alchemy/
            atom_9/  atom_10/  ...  atom_14/     (SDF files)
            dev_target.csv                       (property labels)

Usage:
    python datasets/QM9/eval_ood.py --target 4 --checkpoint pretrained_models/egnn_QM9_gap.pt
    python datasets/QM9/eval_ood.py --target 4 --checkpoint pretrained_models/egnn_QM9_gap.pt --compare_qm9
    python datasets/QM9/eval_ood.py --target 4 --checkpoint pretrained_models/egnn_QM9_gap.pt --size_split

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
    get_qm9_data, build_egnn, build_gat, evaluate_mae,
    QM9_TARGETS, SEED, SchNetWrapper,
)


# ── Constants ───────────────────────────────────────────────────────────────────

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414
# 1 Hartree = 627509.4740631 cal/mol  (for Cv conversion: Eh/K -> cal/mol/K)
EH_TO_CAL_PER_MOL = 627509.4740631

# Alchemy CSV column order (same 12 properties as QM9, same indices)
ALCHEMY_COLUMNS = ['mu', 'alpha', 'homo', 'lumo', 'gap',
                   'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']

# Which targets need Hartree -> eV conversion
HARTREE_TARGETS = {2, 3, 4, 6, 7, 8, 9, 10}  # homo, lumo, gap, zpve, U0, U, H, G
# Cv needs Eh/K -> cal/mol/K
CV_TARGET = 11

# Atom types in QM9 (same one-hot order as PyG's QM9 processing)
QM9_ATOM_TYPES = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}  # H, C, N, O, F
QM9_ALLOWED_Z = set(QM9_ATOM_TYPES.keys())


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

        # build mapping: strip units/quotes from headers, match to QM9 index
        col_map = {}
        for h in headers:
            # extract the short name before any parenthetical unit info
            # handles both "gap" and "gap\n(Ha, LUMO-HOMO)" and '"gap (Ha, ...)"'
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
            # Cv is already in cal/molK in the CSV; no conversion needed

            targets[str(gdb_id).strip()] = y

    print(f'Loaded {len(targets)} target entries from {csv_path}')
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

    # check all atoms are in QM9 vocabulary
    atoms = mol.GetAtoms()
    for atom in atoms:
        if atom.GetAtomicNum() not in QM9_ALLOWED_Z:
            return None

    # get 3D coordinates
    conf = mol.GetConformer()
    pos = torch.tensor(
        [[conf.GetAtomPosition(i).x,
          conf.GetAtomPosition(i).y,
          conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())],
        dtype=torch.float32
    )

    # build node features (11-dim, identical to QM9 in PyG)
    hybridization_map = {
        Chem.rdchem.HybridizationType.SP: [1, 0, 0],
        Chem.rdchem.HybridizationType.SP2: [0, 1, 0],
        Chem.rdchem.HybridizationType.SP3: [0, 0, 1],
    }

    x_list = []
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        # one-hot atom type
        one_hot = [0.0] * 5
        one_hot[QM9_ATOM_TYPES[z]] = 1.0
        # atomic number
        atomic_num = float(z)
        # aromatic
        aromatic = 1.0 if atom.GetIsAromatic() else 0.0
        # hybridization
        hyb = hybridization_map.get(atom.GetHybridization(), [0, 0, 0])
        # number of Hs
        num_hs = float(atom.GetTotalNumHs())

        feat = one_hot + [atomic_num, aromatic] + hyb + [num_hs]
        x_list.append(feat)

    x = torch.tensor(x_list, dtype=torch.float32)

    # build edge_index from bonds (matching QM9 in PyG: undirected bond edges)
    edge_src, edge_dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_src += [i, j]
        edge_dst += [j, i]

    if len(edge_src) == 0:
        return None

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
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

    # targets
    y = targets_dict.get(str(mol_id))
    if y is None:
        return None
    
    z_list = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    z = torch.tensor(z_list, dtype=torch.long)

    # store as (1, 12) to match QM9's data.y shape convention
    data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, y=y.unsqueeze(0), z=z)
    data.num_heavy_atoms = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)

    return data


def load_alchemy_dataset(root, csv_path, max_molecules=None, cache_file=None, force_reload=False):
    """Load Alchemy SDF files and return a list of PyG Data objects, with caching.

    If cache_file exists and force_reload is False, load from cache.
    Otherwise process all SDF files and save to cache_file.
    """
    if cache_file is None:
        cache_file = os.path.join(root, 'processed', 'alchemy_full.pt')

    # Try to load from cache
    if os.path.exists(cache_file) and not force_reload:
        print(f"Loading cached Alchemy dataset from {cache_file}")
        dataset = torch.load(cache_file)
        print(f"Loaded {len(dataset)} molecules from cache.")
        if max_molecules is not None and len(dataset) > max_molecules:
            dataset = dataset[:max_molecules]
        return dataset

    # Otherwise process from scratch
    print("Processing Alchemy SDF files (this may take ~22 min)...")
    targets_dict = load_alchemy_targets(csv_path)

    sdf_files = sorted(glob.glob(os.path.join(root, 'atom_*', '*.sdf')))
    if not sdf_files:
        sdf_files = sorted(glob.glob(os.path.join(root, '*.sdf')))
    if not sdf_files:
        raise FileNotFoundError(f'No SDF files found in {root}.')

    print(f'Found {len(sdf_files)} SDF files.')
    if max_molecules:
        sdf_files = sdf_files[:max_molecules]

    dataset = []
    skipped_atoms = 0
    skipped_parse = 0

    for sdf_path in tqdm(sdf_files, desc='Loading Alchemy'):
        basename = os.path.splitext(os.path.basename(sdf_path))[0]
        mol_id = basename.lstrip('0') or '0'
        data = sdf_to_data(sdf_path, targets_dict, mol_id)
        if data is None:
            data = sdf_to_data(sdf_path, targets_dict, basename)
        if data is None:
            skipped_atoms += 1
            continue
        dataset.append(data)

    print(f'\nAlchemy loaded: {len(dataset)} molecules (QM9-compatible)')
    print(f'Skipped: {skipped_atoms} (non-H/C/N/O/F atoms or missing targets/parse errors)')

    if dataset:
        sizes = [d.num_heavy_atoms for d in dataset]
        from collections import Counter
        dist = Counter(sizes)
        print('Heavy atom distribution:')
        for n in sorted(dist.keys()):
            print(f'  {n} heavy atoms: {dist[n]} molecules')

    # Save to cache
    cache_dir = os.path.dirname(cache_file)
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(dataset, cache_file)
    print(f"Cached Alchemy dataset to {cache_file}")

    return dataset


# ── QM9 size-based OOD (no Alchemy needed) ──────────────────────────────────

def qm9_size_split_ood(target, batch_size=256, seed=SEED):
    """Split QM9 by molecule size for a pseudo-OOD test.

    Train/val: molecules with <= 7 heavy atoms
    OOD test:  molecules with 8-9 heavy atoms

    This doesn't require downloading Alchemy and still tests size generalization.
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

def evaluate_ood(model, loader, target_idx, device):
    """Compute MAE on Out-Of-Distribution (OOD) data for a specific QM9 target."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for data in tqdm(loader, desc='Evaluating OOD', leave=False):
            data = data.to(device)
            out = model(data).squeeze()
            labels = data.y[:, target_idx]
            total_loss += nn.functional.l1_loss(out, labels, reduction='sum').item()
            total_samples += data.num_graphs

    return total_loss / total_samples


def evaluate_per_size(model, dataset, target_idx, device, batch_size=256):
    """Evaluate MAE broken down by number of heavy atoms."""
    from collections import defaultdict

    size_groups = defaultdict(list)
    for d in dataset:
        size_groups[d.num_heavy_atoms].append(d)

    results = {}
    for n_heavy in sorted(size_groups.keys()):
        group = size_groups[n_heavy]
        loader = DataLoader(group, batch_size=batch_size)

        mae = evaluate_ood(model, loader, target_idx, device)
        results[n_heavy] = {'mae': mae, 'count': len(group)}

    return results


# ── Main ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser('QM9 OOD Evaluation (Alchemy)')

    parser.add_argument('--model',      type=str, default='egnn', choices=['egnn', 'gat', 'schnet'], help='Choose a model to evaluate')
    parser.add_argument('--target',     type=int, default=4,
                        help='QM9 target index (0-11). Default: 4 (gap)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to pretrained QM9 model checkpoint')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed',       type=int, default=SEED)

    # Alchemy options
    parser.add_argument('--alchemy_root', type=str, default='data/Alchemy',
                        help='Root dir with atom_N/ subdirectories of SDF files')
    parser.add_argument('--alchemy_csv',  type=str, default='data/Alchemy/final_version.csv',
                        help='Path to Alchemy target CSV')
    parser.add_argument('--max_molecules', type=int, default=None,
                        help='Cap the number of Alchemy molecules to load (for quick testing)')
    
    # use cached version of alchemy dataset for ood evaluation
    parser.add_argument('--alchemy_cache', type=str,
                    default='data/Alchemy/processed/alchemy_full.pt',
                    help='Path to cache the processed Alchemy dataset (if not exists, it will be created)')
    parser.add_argument('--force_reload', action='store_true',
                        help='Force reprocessing of Alchemy dataset even if cache exists')

    # Comparison options
    parser.add_argument('--compare_qm9', action='store_true',
                        help='Also evaluate on QM9 test set for side-by-side comparison')
    parser.add_argument('--size_split',  action='store_true',
                        help='Use QM9 size-based OOD split instead of Alchemy '
                             '(no extra download needed)')

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    target_name, target_unit = QM9_TARGETS[args.target]
    print(f'\n--- OOD Evaluation ---')
    print(f'Target {args.target}: {target_name} [{target_unit}]')

    # load model
    state = load_checkpoint(args.checkpoint, device=str(device))
    hparams = state.get('hparams', {})

    match args.model:
        case 'egnn': model = build_egnn(in_dim=hparams.get('in_dim', 11),
                                        hid_dim=hparams.get('hid_dim', 64),
                                        out_dim=1,
                                        n_layers=hparams.get('n_layers', 4))
        
        case 'gat': model = build_gat(in_dim=hparams.get('in_dim', 14),
                                      hid_dim=hparams.get('hid_dim', 64),
                                      out_dim=1,
                                      n_layers=hparams.get('n_layers', 4),
                                      n_heads=hparams.get('n_heads', 4))
            
        case 'schnet': model = SchNetWrapper(hidden_channels=hparams.get('hidden_channels', 64),
                                      num_filters=hparams.get('num_filters', 64),
                                      num_interactions=hparams.get('num_interactions', 8),
                                      num_gaussians=hparams.get('num_gaussians', 25),
                                      cutoff=hparams.get('cutoff', 10.0))
            
        case _ : model = build_egnn(in_dim=hparams.get('in_dim', 11),
                                    hid_dim=hparams.get('hid_dim', 64),
                                    out_dim=1,
                                    n_layers=hparams.get('n_layers', 4))
            

    model.load_state_dict(state['state_dict'])
    model.to(device)
    model.eval()

    # ── In-distribution baseline (QM9 test set) ────────────────────────────
    qm9_test_mae = None
    if args.compare_qm9 or args.size_split:
        _, _, qm9_test_loader, _ = get_qm9_data(
            target=args.target, batch_size=args.batch_size, seed=args.seed
        )
        qm9_test_mae = evaluate_ood(model, qm9_test_loader, args.target, device)
        print(f'\nQM9 in-distribution test MAE: {qm9_test_mae:.4f} {target_unit}')

    # ── QM9 size-based OOD ──────────────────────────────────────────────────
    if args.size_split:
        print(f'\n--- QM9 Size-based OOD ---')
        ood_loader, n_ood = qm9_size_split_ood(
            args.target, batch_size=args.batch_size, seed=args.seed
        )
        size_ood_mae = evaluate_ood(model, ood_loader, args.target, device)
        print(f'QM9 large-molecule OOD MAE: {size_ood_mae:.4f} {target_unit}  ({n_ood} molecules)')

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
        alchemy_data = load_alchemy_dataset(args.alchemy_root, args.alchemy_csv,
                                            max_molecules=args.max_molecules,
                                            cache_file=args.alchemy_cache,
                                            force_reload=args.force_reload
                                        )

        if not alchemy_data:
            print('No compatible Alchemy molecules loaded.')
            return

        alchemy_loader = DataLoader(alchemy_data, batch_size=args.batch_size)

        alchemy_mae = evaluate_ood(model, alchemy_loader, args.target, device)

        print(f'\n{"="*60}')
        print(f'Target: {target_name} [{target_unit}]')
        print(f'{"="*60}')
        if qm9_test_mae is not None:
            print(f'QM9 test MAE (in-distribution):  {qm9_test_mae:.4f} {target_unit}')
        print(f'Alchemy MAE (out-of-distribution): {alchemy_mae:.4f} {target_unit}')
        print(f'Alchemy molecules evaluated: {len(alchemy_data)}')
        if qm9_test_mae is not None and qm9_test_mae > 0:
            ratio = alchemy_mae / qm9_test_mae
            print(f'OOD / in-distribution ratio: {ratio:.2f}x')
        print(f'{"="*60}')

        # per-size breakdown
        print(f'\nPer-size breakdown:')
        size_results = evaluate_per_size(
            model, alchemy_data, args.target, device, args.batch_size
        )
        print(f'{"Heavy atoms":<14} {"Count":>8} {"MAE":>12}')
        print(f'{"─"*36}')
        for n_heavy, res in size_results.items():
            print(f'{n_heavy:<14} {res["count"]:>8} {res["mae"]:>12.4f}')

        # save results
        os.makedirs('results/csv', exist_ok=True)
        model_name = state.get('model_class', 'unknown')
        csv_path = f'results/csv/ood_{model_name}_alchemy_{target_name}.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['source', 'n_heavy_atoms', 'count', 'mae', 'unit'])
            if qm9_test_mae is not None:
                writer.writerow(['QM9_test', 'all', '', f'{qm9_test_mae:.6f}', target_unit])
            writer.writerow(['Alchemy_all', 'all', len(alchemy_data),
                             f'{alchemy_mae:.6f}', target_unit])
            for n_heavy, res in size_results.items():
                writer.writerow([f'Alchemy_{n_heavy}', n_heavy, res['count'],
                                 f'{res["mae"]:.6f}', target_unit])
        print(f'\nResults saved: {csv_path}')


if __name__ == '__main__':
    main()
