"""
Training script for energy prediction on (revised) MD17.
TODO: Currently implementation is incomplete and models are not
tested with MD17
Usage:
    python datasets/MD17/train.py --model egnn --molecule "revised aspirin"
    python datasets/MD17/train.py --model egnn --molecule "revised ethanol" --epochs 200 --hid_dim 128
    python datasets/MD17/train.py --model egnn --train_all_molecules

Run from the project root (GNN-molecular/).
"""

import os
import sys
import time
import copy
import argparse
import csv

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.util import (
    set_seed, get_device, count_parameters,
    get_md17_data, save_checkpoint,
    evaluate_mae_energy, RMD17_MOLECULES, SEED,
)
from utils.metrics_tracker import MetricsTracker
from utils.util import SchNetWrapper


def parse_args():
    parser = argparse.ArgumentParser('MD17 Training')

    # model
    parser.add_argument('--model',    default='egnn', choices=['egnn'])
    parser.add_argument('--hid_dim',  type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=4)

    # data
    parser.add_argument('--molecule',   type=str, default='aspirin',
                        help='MD17 molecule name (e.g. "revised aspirin")')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_size', type=int, default=1000)
    parser.add_argument('--val_size',   type=int, default=1000)
    parser.add_argument('--cutoff',     type=float, default=5.0,
                        help='Radius graph cutoff in Angstrom')

    # training
    parser.add_argument('--epochs',       type=int,   default=200)
    parser.add_argument('--lr',           type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience',     type=int,   default=30, help='Early stopping patience')
    parser.add_argument('--lr_patience',  type=int,   default=10)
    parser.add_argument('--lr_factor',    type=float, default=0.5)
    parser.add_argument('--step',         type=int,   default=1)
    parser.add_argument('--seed',         type=int,   default=SEED)

    # sweep
    parser.add_argument('--train_all_molecules', action='store_true',
                        help='Train on all revised MD17 molecules sequentially')

    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, mean, std, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    train_bar = tqdm(loader, desc='[Train]', leave=False)
    for data in train_bar:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data).squeeze()
        loss = nn.functional.l1_loss(out, (data.energy.squeeze() - mean) / std)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs
        train_bar.set_postfix(loss=f'{loss.item():.4f}')

    return (total_loss / total_samples) * std


def molecule_tag(molecule):
    """Convert molecule name to a filesystem-safe tag."""
    return molecule.replace(' ', '_').replace('revised_', 'r')


def train_model(args, model, train_loader, val_loader, mean, std, device, mol_name):
    tag = molecule_tag(mol_name)
    run_name = f'{args.model}_MD17_{tag}'

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience
    )

    metrics = MetricsTracker(run_name)

    best_val_mae = float('inf')
    best_state = None
    best_epoch = 0
    no_improve = 0

    print(f'\n{"="*60}')
    print(f'Training: {run_name}')
    print(f'Molecule: {mol_name}')
    print(f'Parameters: {count_parameters(model):,}')
    print(f'{"="*60}\n')

    total_start = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_mae = train_one_epoch(model, train_loader, optimizer, mean, std, device)
        val_mae = evaluate_mae_energy(model, val_loader, mean, std, device)

        scheduler.step(val_mae)
        lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        if epoch % args.step == 0:
            metrics.update(epoch, train_mae, val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = {
                'epoch': epoch,
                'state_dict': copy.deepcopy(model.state_dict()),
                'optimizer': copy.deepcopy(optimizer.state_dict()),
                'val_mae': val_mae,
            }
            no_improve = 0
        else:
            no_improve += 1

        print(
            f'Epoch {epoch:>3} | Train MAE: {train_mae:.6f} | Val MAE: {val_mae:.6f} | '
            f'LR: {lr:.1e} | Best: {best_val_mae:.6f} | Time: {epoch_time:.1f}s'
        )

        if no_improve >= args.patience:
            print(f'Early stopping at epoch {epoch} (patience={args.patience})')
            break

    total_time = time.time() - total_start
    print(f'\nTraining complete: {total_time/60:.1f} min')
    print(f'Best val MAE: {best_val_mae:.6f} [kcal/mol] at epoch {best_epoch}')

    if best_state is not None:
        model.load_state_dict(best_state['state_dict'])

    save_checkpoint(
        state={
            'name': run_name,
            'model_class': args.model,
            'molecule': mol_name,
            'epoch': best_state['epoch'] if best_state else epoch,
            'state_dict': model.state_dict(),
            'hparams': vars(args),
            'val_mae': best_val_mae,
            'seed': args.seed,
        },
        save_dir='pretrained_models'
    )

    metrics.save()
    return metrics, best_val_mae


def run_single_molecule(args, device):
    """Train on a single molecule and evaluate."""
    use_cuda = device.type == 'cuda'

    train_loader, val_loader, test_loader, (mean, std), in_dim = get_md17_data(
        molecule=args.molecule, batch_size=args.batch_size,
        train_size=args.train_size, val_size=args.val_size,
        cutoff=args.cutoff, seed=args.seed,
    )

    match args.model:
        case _ : model = SchNetWrapper(
        hidden_channels=64,
        num_filters=64,
        num_interactions=8,
        num_gaussians=25,
        cutoff=10.0,
    )

    metrics, best_val = train_model(args, model, train_loader, val_loader, mean, std, device, args.molecule)

    test_mae = evaluate_mae_energy(model, test_loader, mean, std, device)
    print(f'\nTest MAE: {test_mae:.6f} [kcal/mol]  (molecule: {args.molecule})')

    return test_mae, best_val


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    print(f'\n--- MD17 Energy Prediction ---')

    if args.train_all_molecules:
        results = []
        for mol in RMD17_MOLECULES:
            print(f'\n{"#"*60}')
            print(f'# Molecule: {mol}')
            print(f'{"#"*60}')
            args.molecule = mol
            test_mae, best_val = run_single_molecule(args, device)
            results.append({
                'molecule': mol,
                'test_mae': test_mae,
                'best_val_mae': best_val,
            })

        # summary
        print(f'\n{"="*55}')
        print(f'{"Molecule":<28} {"Test MAE":>12} {"Val MAE":>12}')
        print(f'{"─"*55}')
        for r in results:
            print(f'{r["molecule"]:<28} {r["test_mae"]:>12.6f} {r["best_val_mae"]:>12.6f}')
        print(f'{"="*55}')

        # save
        os.makedirs('results/csv', exist_ok=True)
        csv_path = f'results/csv/md17_{args.model}_all_molecules.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['molecule', 'test_mae', 'best_val_mae'])
            writer.writeheader()
            writer.writerows(results)
        print(f'Results saved: {csv_path}')

    else:
        test_mae, best_val = run_single_molecule(args, device)

        os.makedirs('results/csv', exist_ok=True)
        tag = molecule_tag(args.molecule)
        csv_path = f'results/csv/md17_{args.model}_{tag}.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['model', 'molecule', 'test_mae', 'best_val_mae',
                             'hid_dim', 'n_layers', 'train_size', 'seed'])
            writer.writerow([args.model, args.molecule, f'{test_mae:.6f}', f'{best_val:.6f}',
                             args.hid_dim, args.n_layers, args.train_size, args.seed])
        print(f'Results saved: {csv_path}')


if __name__ == '__main__':
    main()
