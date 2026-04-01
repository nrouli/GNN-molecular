"""
Training script for molecular property prediction on QM9.
Supports EGNN (and future models) with configurable targets.

Usage:
    python datasets/QM9/train.py --model egnn --target 0 --epochs 50
    python datasets/QM9/train.py --model egnn --target 7 --epochs 100 --hid_dim 64

Run from the project root (GNN-molecular/).
"""

import os
import sys
import time
import copy
import argparse

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from torch_geometric.nn import SchNet

# project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.util import (
    set_seed, get_device, count_parameters,
    get_qm9_data, build_egnn, build_gat, save_checkpoint,
    evaluate_mae, SchNetWrapper, QM9_TARGETS, SEED,
)

from utils.benchmark import benchmark_inference
from utils.metrics_tracker import MetricsTracker


def parse_args():
    parser = argparse.ArgumentParser('QM9 Training')

    # model
    parser.add_argument('--model',    default='egnn', choices=['egnn', 'schnet', 'gat'], help='Model architecture')
    parser.add_argument('--in_dim',   type=int, default=14,  help='Node feature dimension')
    parser.add_argument('--hid_dim',  type=int, default=64,  help='Hidden dimension')
    parser.add_argument('--n_layers', type=int, default=4,   help='Number of message passing layers')
    
    # gat params
    parser.add_argument('--n_heads',  type=int, default=4,   help='Number of attention heads for GAT')
    
    # schnet params
    parser.add_argument('--hidden_channels',  type=int,   default=64,   help='(SchNet) Number of hidden channels')
    parser.add_argument('--num_filters',      type=int,   default=64,   help='(SchNet) Number of filters')
    parser.add_argument('--num_interactions', type=int,   default=8,    help='(SchNet) Number of interactions')
    parser.add_argument('--num_gaussians',    type=int,   default=25,   help='(SchNet) Number of Gaussians')
    parser.add_argument('--cutoff',           type=float, default=10.0, help='(SchNet) Cutoff distance (in Angstroms)')


    # data
    parser.add_argument('--target',     type=int, default=4,   help='QM9 target index (0-11)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')

    # training
    parser.add_argument('--epochs',       type=int,   default=200,   help='Max epochs')
    parser.add_argument('--lr',           type=float, default=1e-3,  help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-16, help='Weight decay')
    parser.add_argument('--patience',     type=int,   default=20,    help='Early stopping patience')
    parser.add_argument('--lr_patience',  type=int,   default=5,     help='LR scheduler patience')
    parser.add_argument('--lr_factor',    type=float, default=0.5,   help='LR reduction factor')
    parser.add_argument('--step',         type=int,   default=1,     help='Logging interval (epochs)')
    parser.add_argument('--seed',         type=int,   default=SEED,  help='Random seed')

    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, target_idx, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    train_bar = tqdm(loader, desc='[Train]', leave=False)
    for data in train_bar:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data).squeeze()
        loss = nn.functional.l1_loss(out, data.y[:, target_idx])
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs
        train_bar.set_postfix(loss=f'{loss.item():.4f}')

    return total_loss / total_samples


def train_model(args, model, train_loader, val_loader, device):
    """training loop"""
    
    target_name, target_unit = QM9_TARGETS[args.target]
    run_name = f'{args.model}_QM9_{target_name}'

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
    print(f'Parameters: {count_parameters(model):,}')
    print(f'{"="*60}\n')

    total_start = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # train
        train_mae = train_one_epoch(model, train_loader, optimizer, args.target, device)

        # validate
        val_mae = evaluate_mae(model, val_loader, args.target, device)

        scheduler.step(val_mae)
        lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        # track metrics (MetricsTracker expects loss; MAE serves as both loss and score here)
        if epoch % args.step == 0:
            metrics.update(epoch, train_mae, val_mae)

        # early stopping
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
            f'Epoch {epoch:>3} | Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f} | '
            f'LR: {lr:.1e} | Best: {best_val_mae:.4f} | Time: {epoch_time:.1f}s'
        )

        if no_improve >= args.patience:
            print(f'Early stopping at epoch {epoch} (patience={args.patience})')
            break

    total_time = time.time() - total_start
    print(f'\nTraining complete: {total_time/60:.1f} min')
    print(f'Best val MAE: {best_val_mae:.4f} [{target_unit}] at epoch {best_epoch}')

    # restore best weights
    if best_state is not None:
        model.load_state_dict(best_state['state_dict'])

    # save checkpoint
    save_checkpoint(
        state={
            'name': run_name,
            'model_class': args.model,
            'epoch': best_state['epoch'] if best_state else epoch,
            'state_dict': model.state_dict(),
            'hparams': vars(args),
            'val_mae': best_val_mae,
            'seed': args.seed,
        },
        save_dir='pretrained_models'
    )

    # save metrics
    metrics.save()

    return metrics, best_val_mae


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    use_cuda = device.type == 'cuda'

    print(f'\n--- QM9 Property Prediction ---')

    # data
    train_loader, val_loader, test_loader, (mean, std) = get_qm9_data(
        target=args.target, batch_size=args.batch_size
    )

    # model
    match args.model:
        case 'egnn': model = build_egnn(
            in_dim=args.in_dim, hid_dim=args.hid_dim,
            out_dim=1, n_layers=args.n_layers
        )
        case 'schnet': model = SchNetWrapper(hidden_channels=args.hidden_channels,
                                      num_filters=args.num_filters,
                                      num_interactions=args.num_interactions,
                                      num_gaussians=args.num_gaussians,
                                      cutoff=args.cutoff)
        case 'gat': model = build_gat(in_dim=args.in_dim, hid_dim=args.hid_dim, out_dim=1, n_layers=args.n_layers, n_heads=args.n_heads)
        case _: model = build_egnn(
            in_dim=args.in_dim, hid_dim=args.hid_dim,
            out_dim=1, n_layers=args.n_layers
        )


    model.to(device)
    # train
    metrics, best_val = train_model(args, model, train_loader, val_loader, device)

    # final test evaluation
    test_mae = evaluate_mae(model, test_loader, args.target, device)
    target_name, target_unit = QM9_TARGETS[args.target]
    print(f'\nTest MAE: {test_mae:.4f} [{target_unit}]  (target: {target_name})')

    # save test result
    os.makedirs('results/csv', exist_ok=True)
    import csv
    csv_path = f'results/csv/qm9_{args.model}_{target_name}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'target', 'target_name', 'test_mae', 'unit', 'best_val_mae',
                         'hid_dim', 'n_layers', 'epochs_trained', 'seed'])
        writer.writerow([args.model, args.target, target_name, f'{test_mae:.6f}', target_unit,
                         f'{best_val:.6f}', args.hid_dim, args.n_layers, args.epochs, args.seed])
    print(f'Results saved: {csv_path}')


if __name__ == '__main__':
    main()

