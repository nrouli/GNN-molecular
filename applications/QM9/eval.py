"""
Evaluation script for trained models on QM9.

Supports:
  - Single-target evaluation with detailed statistics
  - Multi-target sweep (all 12 QM9 properties)
  - Multiple trials for confidence intervals

Usage:
    python datasets/QM9/eval.py --model egnn --target 0
    python datasets/QM9/eval.py --model egnn --target 0 --trials 5
    python datasets/QM9/eval.py --model egnn --sweep_all

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
    get_qm9_data, build_egnn, evaluate_mae,
    QM9_TARGETS, SEED,
)


def parse_args():
    parser = argparse.ArgumentParser('QM9 Evaluation')

    parser.add_argument('--model',      default='egnn', choices=['egnn'], help='Model architecture')
    parser.add_argument('--target',     type=int, default=0, help='QM9 target index (0-11)')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--trials',     type=int, default=1, help='Repeated evals (>1 useful if loader is stochastic)')
    parser.add_argument('--seed',       type=int, default=SEED)

    parser.add_argument('--sweep_all',  action='store_true',
                        help='Evaluate all 12 QM9 targets (requires a checkpoint per target)')

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint. If omitted, auto-resolves from pretrained_models/')

    return parser.parse_args()


def resolve_checkpoint(model_name, target_name, ckpt_dir='pretrained_models'):
    """Try to find a checkpoint following the naming convention."""
    path = os.path.join(ckpt_dir, f'{model_name}_QM9_{target_name}.pt')
    if os.path.isfile(path):
        return path
    # fallback: try .tar
    path_tar = os.path.join(ckpt_dir, f'{model_name}_QM9_{target_name}.tar')
    if os.path.isfile(path_tar):
        return path_tar
    return None


def load_model_from_checkpoint(ckpt_path, device):
    """Load model weights from a checkpoint."""
    state = load_checkpoint(ckpt_path, device=str(device))
    hparams = state.get('hparams', {})

    # reconstruct model
    model = build_egnn(
        in_dim=hparams.get('in_dim', 11),
        hid_dim=hparams.get('hid_dim', 128),
        out_dim=1,
        n_layers=hparams.get('n_layers', 4)
    )
    model.load_state_dict(state['state_dict'])
    model.to(device)
    return model, hparams


def evaluate_single_target(args, device):
    """Evaluate a single QM9 target and print results."""
    target_name, target_unit = QM9_TARGETS[args.target]

    # resolve checkpoint
    ckpt_path = args.checkpoint or resolve_checkpoint(args.model, target_name)
    if ckpt_path is None:
        print(f'No checkpoint found for {args.model} / {target_name}. Skipping.')
        return None

    model, hparams = load_model_from_checkpoint(ckpt_path, device)

    # data (same split as training thanks to fixed seed)
    _, _, test_loader, (mean, std) = get_qm9_data(
        target=args.target, batch_size=args.batch_size, seed=args.seed
    )

    # evaluate
    maes = []
    for trial in range(args.trials):
        mae = evaluate_mae(model, test_loader, args.target, device)
        maes.append(mae)

    maes = np.array(maes)
    print(f'\n{"─"*50}')
    print(f'Target {args.target}: {target_name} [{target_unit}]')
    print(f'Model:  {args.model} ({count_parameters(model):,} params)')
    print(f'Checkpoint: {ckpt_path}')
    print(f'{"─"*50}')
    if args.trials == 1:
        print(f'Test MAE: {maes[0]:.4f} {target_unit}')
    else:
        print(f'Test MAE: {maes.mean():.4f} ± {maes.std():.4f} {target_unit}  ({args.trials} trials)')
    print()

    return {
        'target': args.target,
        'target_name': target_name,
        'unit': target_unit,
        'mae_mean': maes.mean(),
        'mae_std': maes.std(),
    }


def sweep_all_targets(args, device):
    """Evaluate all 12 QM9 targets and produce a summary table."""
    results = []

    for target_idx in range(12):
        target_name, target_unit = QM9_TARGETS[target_idx]
        ckpt_path = resolve_checkpoint(args.model, target_name)
        if ckpt_path is None:
            print(f'[SKIP] No checkpoint for target {target_idx} ({target_name})')
            continue

        model, hparams = load_model_from_checkpoint(ckpt_path, device)
        _, _, test_loader, _ = get_qm9_data(
            target=target_idx, batch_size=args.batch_size, seed=args.seed
        )

        mae = evaluate_mae(model, test_loader, target_idx, device)
        results.append({
            'target': target_idx,
            'name': target_name,
            'unit': target_unit,
            'test_mae': mae,
        })
        print(f'  {target_name:<6} : {mae:.4f} {target_unit}')

    # summary table
    print(f'\n{"="*55}')
    print(f'{"Target":<8} {"Name":<6} {"MAE":>10} {"Unit":<12}')
    print(f'{"─"*55}')
    for r in results:
        print(f'{r["target"]:<8} {r["name"]:<6} {r["test_mae"]:>10.4f} {r["unit"]:<12}')
    print(f'{"="*55}')

    # save to csv
    os.makedirs('results/csv', exist_ok=True)
    csv_path = f'results/csv/qm9_{args.model}_sweep.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['target', 'name', 'unit', 'test_mae'])
        writer.writeheader()
        writer.writerows(results)
    print(f'Sweep results saved: {csv_path}')

    return results


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    print(f'\n--- QM9 Evaluation ---')

    if args.sweep_all:
        sweep_all_targets(args, device)
    else:
        result = evaluate_single_target(args, device)


if __name__ == '__main__':
    main()
