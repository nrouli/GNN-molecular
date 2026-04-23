"""
Evaluation script for trained models on (revised) MD17.

Supports:
  - Single-molecule evaluation
  - All-molecule sweep
  - Multiple trials for confidence intervals

Usage:
    python datasets/MD17/eval.py --model egnn --molecule "revised aspirin"
    python datasets/MD17/eval.py --model egnn --molecule "revised aspirin" --trials 5
    python datasets/MD17/eval.py --model egnn --sweep_all

Run from the project root (GNN-molecular/).
"""

import os
import sys
import argparse
import csv

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.util import (
    set_seed, get_device, count_parameters, load_checkpoint,
    get_md17_data, build_egnn, evaluate_mae_energy,
    RMD17_MOLECULES, SEED,
)


def parse_args():
    parser = argparse.ArgumentParser('MD17 Evaluation')

    parser.add_argument('--model',      default='egnn', choices=['egnn'])
    parser.add_argument('--molecule',   type=str, default='revised aspirin')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--cutoff',     type=float, default=5.0)
    parser.add_argument('--trials',     type=int, default=1)
    parser.add_argument('--seed',       type=int, default=SEED)

    parser.add_argument('--sweep_all',  action='store_true',
                        help='Evaluate all revised MD17 molecules')
    parser.add_argument('--checkpoint', type=str, default=None)

    return parser.parse_args()


def molecule_tag(molecule):
    return molecule.replace(' ', '_').replace('revised_', 'r')


def resolve_checkpoint(model_name, molecule, ckpt_dir='pretrained_models'):
    tag = molecule_tag(molecule)
    for ext in ['.pt', '.tar']:
        path = os.path.join(ckpt_dir, f'{model_name}_MD17_{tag}{ext}')
        if os.path.isfile(path):
            return path
    return None


def load_model_from_checkpoint(ckpt_path, device):
    state = load_checkpoint(ckpt_path, device=str(device))
    hparams = state.get('hparams', {})

    model = build_egnn(
        in_dim=hparams.get('in_dim', 10),
        hid_dim=hparams.get('hid_dim', 128),
        out_dim=1,
        n_layers=hparams.get('n_layers', 4)
    )
    model.load_state_dict(state['state_dict'])
    model.to(device)
    return model, hparams


def evaluate_single_molecule(args, device):
    ckpt_path = args.checkpoint or resolve_checkpoint(args.model, args.molecule)
    if ckpt_path is None:
        print(f'No checkpoint found for {args.model} / {args.molecule}. Skipping.')
        return None

    model, hparams = load_model_from_checkpoint(ckpt_path, device)

    _, _, test_loader, (mean, std) = get_md17_data(
        molecule=args.molecule, batch_size=args.batch_size,
        train_size=args.train_size, val_size=args.val_size,
        cutoff=args.cutoff, seed=args.seed,
    )

    maes = []
    for _ in range(args.trials):
        mae = evaluate_mae_energy(model, test_loader, device)
        maes.append(mae)

    maes = np.array(maes)
    tag = molecule_tag(args.molecule)

    print(f'\n{"─"*55}')
    print(f'Molecule:   {args.molecule}')
    print(f'Model:      {args.model} ({count_parameters(model):,} params)')
    print(f'Checkpoint: {ckpt_path}')
    print(f'{"─"*55}')
    if args.trials == 1:
        print(f'Test MAE: {maes[0]:.6f} [kcal/mol]')
    else:
        print(f'Test MAE: {maes.mean():.6f} ± {maes.std():.6f} [kcal/mol]  ({args.trials} trials)')
    print()

    return {
        'molecule': args.molecule,
        'mae_mean': maes.mean(),
        'mae_std': maes.std(),
    }


def sweep_all_molecules(args, device):
    results = []

    for mol in RMD17_MOLECULES:
        ckpt_path = resolve_checkpoint(args.model, mol)
        if ckpt_path is None:
            print(f'[SKIP] No checkpoint for {mol}')
            continue

        model, hparams = load_model_from_checkpoint(ckpt_path, device)

        _, _, test_loader, _ = get_md17_data(
            molecule=mol, batch_size=args.batch_size,
            train_size=args.train_size, val_size=args.val_size,
            cutoff=args.cutoff, seed=args.seed,
        )

        mae = evaluate_mae_energy(model, test_loader, device)
        results.append({'molecule': mol, 'test_mae': mae})
        print(f'  {mol:<28} : {mae:.6f} kcal/mol')

    # summary
    print(f'\n{"="*50}')
    print(f'{"Molecule":<28} {"Test MAE":>12}')
    print(f'{"─"*50}')
    for r in results:
        print(f'{r["molecule"]:<28} {r["test_mae"]:>12.6f}')
    print(f'{"="*50}')

    os.makedirs('results/csv', exist_ok=True)
    csv_path = f'results/csv/md17_{args.model}_eval_sweep.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['molecule', 'test_mae'])
        writer.writeheader()
        writer.writerows(results)
    print(f'Results saved: {csv_path}')

    return results


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    print(f'\n--- MD17 Evaluation ---')

    if args.sweep_all:
        sweep_all_molecules(args, device)
    else:
        evaluate_single_molecule(args, device)


if __name__ == '__main__':
    main()
