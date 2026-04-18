"""
Training script for molecular property prediction on QM9.
Supports EGNN (and future models) with configurable targets.

Usage:
    python applications/QM9/train.py --model egnn --target 0 --epochs 50
    python applications/QM9/train.py --model egnn --target 4 --resume   # resume from last checkpoint

Run from the project root (GNN-molecular/).
"""

import os
import sys
import time
import copy
import argparse
import glob

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from torch_geometric.nn import SchNet

# project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.util import (
    set_seed, get_device, count_parameters,
    get_qm9_data, build_egnn, build_gat, build_ga_gat, build_cgenn, save_checkpoint,
    evaluate_mae, SchNetWrapper, QM9_TARGETS, SEED,
)

from utils.benchmark import benchmark_inference
from utils.metrics_tracker import MetricsTracker


# ── Checkpoint helpers ───────────────────────────────────────────────

def get_checkpoint_dir(model_name, target_name):
    """pretrained_models/checkpoints/<model>_<target>/"""
    return os.path.join('pretrained_models', 'checkpoints', f'{model_name}_{target_name}')


def save_training_checkpoint(state, model_name, target_name, epoch):
    """Save a periodic training checkpoint, keeping only the last 3."""
    ckpt_dir = get_checkpoint_dir(model_name, target_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f'ckpt_epoch_{epoch:04d}.pt')
    torch.save(state, path)

    existing = sorted(glob.glob(os.path.join(ckpt_dir, 'ckpt_epoch_*.pt')))

    print(f'  [Checkpoint saved: {path}]')


def find_latest_checkpoint(model_name, target_name):
    """Return path to the most recent checkpoint, or None."""
    ckpt_dir = get_checkpoint_dir(model_name, target_name)
    files = sorted(glob.glob(os.path.join(ckpt_dir, 'ckpt_epoch_*.pt')))
    return files[-1] if files else None


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser('QM9 Training')

    # model
    parser.add_argument('--model',    default='egnn', choices=['egnn', 'schnet', 'gat', 'gagat', 'cgenn'])
    parser.add_argument('--in_dim',   type=int, default=11)
    parser.add_argument('--hid_dim',  type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=4)

    # gat
    parser.add_argument('--n_heads',  type=int, default=4)

    # schnet
    parser.add_argument('--hidden_channels',  type=int,   default=128)
    parser.add_argument('--num_filters',      type=int,   default=128)
    parser.add_argument('--num_interactions', type=int,   default=6)
    parser.add_argument('--num_gaussians',    type=int,   default=25)
    parser.add_argument('--cutoff',           type=float, default=10.0)

    # data
    parser.add_argument('--target',     type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--train_size', type=int, default=100000)
    parser.add_argument('--val_size',   type=int, default=10000)
    parser.add_argument('--test_size',  type=int, default=20000)
    # training
    parser.add_argument('--epochs',       type=int,   default=200)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--patience',     type=int,   default=20)
    parser.add_argument('--lr_patience',  type=int,   default=5)
    parser.add_argument('--lr_factor',    type=float, default=0.5)
    parser.add_argument('--step',         type=int,   default=1)
    parser.add_argument('--seed',         type=int,   default=SEED)

    # checkpointing
    parser.add_argument('--ckpt_every',   type=int,   default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume',       action='store_true',
                        help='Resume training from the latest checkpoint')

    return parser.parse_args()


# ── Training ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, target_idx, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    train_bar = tqdm(loader, desc='[Train]', leave=False)
    for data in train_bar:
        data = data.to(device)
        optimizer.zero_grad()

        with torch.autocast('cuda', dtype=torch.float16):
            pred = model(data)
            out = pred.squeeze()
            loss = nn.functional.l1_loss(out, data.y[:, target_idx])

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs
        train_bar.set_postfix(loss=f'{loss.item():.4f}')

    return total_loss / total_samples


def train_model(args, model, train_loader, val_loader, device):
    """Training loop with periodic checkpointing and resume."""

    target_name, target_unit = QM9_TARGETS[args.target]
    run_name = f'{args.model}_QM9_{target_name}'

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience
    )
    scaler = torch.amp.GradScaler()
    metrics = MetricsTracker(run_name)

    best_val_mae = float('inf')
    best_state = None
    best_epoch = 0
    no_improve = 0
    start_epoch = 0

    # ── Resume ──
    if args.resume:
        ckpt_path = find_latest_checkpoint(args.model, target_name)
        if ckpt_path is not None:
            print(f'\nResuming from: {ckpt_path}')
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            scaler.load_state_dict(ckpt['scaler'])
            start_epoch  = ckpt['epoch'] + 1
            best_val_mae = ckpt['best_val_mae']
            best_epoch   = ckpt['best_epoch']
            no_improve   = ckpt['no_improve']
            if ckpt.get('best_state_dict') is not None:
                best_state = {
                    'epoch': best_epoch,
                    'state_dict': ckpt['best_state_dict'],
                    'optimizer': copy.deepcopy(optimizer.state_dict()),
                    'val_mae': best_val_mae,
                }
            print(f'  Epoch {start_epoch}, best val MAE so far: {best_val_mae:.4f}\n')
        else:
            print('\nNo checkpoint found; starting from scratch.\n')

    print(f'{"="*60}')
    print(f'Training: {run_name}')
    print(f'Parameters: {count_parameters(model):,}')
    print(f'{"="*60}\n')

    total_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        train_mae = train_one_epoch(model, train_loader, optimizer, scaler,
                                    args.target, device)
        val_mae = evaluate_mae(model, val_loader, args.target, device)

        scheduler.step(val_mae)
        lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        if epoch % args.step == 0:
            metrics.update(epoch, train_mae, val_mae)

        # best tracking
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

        # periodic checkpoint
        if (epoch + 1) % args.ckpt_every == 0:
            save_training_checkpoint(
                state={
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_val_mae': best_val_mae,
                    'best_epoch': best_epoch,
                    'best_state_dict': best_state['state_dict'] if best_state else None,
                    'no_improve': no_improve,
                    'hparams': vars(args),
                },
                model_name=args.model,
                target_name=target_name,
                epoch=epoch,
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

    # save final model
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

    metrics.save()
    return metrics, best_val_mae


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    print(f'\n--- QM9 Property Prediction ---')

    train_loader, val_loader, test_loader, (mean, std) = get_qm9_data(
        target=args.target, batch_size=args.batch_size,
        train_size=args.train_size, 
        val_size=args.val_size,
        test_size=args.test_size
    )

    # build model
    match args.model:
        case 'egnn':
            model = build_egnn(in_dim=args.in_dim, hid_dim=args.hid_dim,
                               out_dim=1, n_layers=args.n_layers)
        case 'schnet':
            model = SchNetWrapper(hidden_channels=args.hidden_channels,
                                  num_filters=args.num_filters,
                                  num_interactions=args.num_interactions,
                                  num_gaussians=args.num_gaussians,
                                  cutoff=args.cutoff)
        case 'gat':
            model = build_gat(in_dim=args.in_dim, hid_dim=args.hid_dim,
                              out_dim=1, n_layers=args.n_layers, n_heads=args.n_heads)
        case 'gagat':
            model = build_ga_gat(in_dim=args.in_dim, hid_dim=args.hid_dim,
                                out_dim=1, n_layers=args.n_layers, n_heads=args.n_heads)
        case 'cgenn':
            model = build_cgenn(in_dim=args.in_dim, hid_dim=args.hid_dim,
                                out_dim=1, n_layers=args.n_layers)
        case _:
            model = build_egnn(in_dim=args.in_dim, hid_dim=args.hid_dim,
                               out_dim=1, n_layers=args.n_layers)

    model.to(device)

    # profiling and benchmarking only on fresh runs
    if not args.resume:
        from torch.profiler import profile, ProfilerActivity

        batch = next(iter(train_loader)).to('cuda')
        model.eval()
        with torch.no_grad(), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            _ = model(batch)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        benchmark_batch = next(iter(train_loader))
        benchmark_inference(model=model, input_data=benchmark_batch, device='cuda')

    # train
    metrics, best_val = train_model(args, model, train_loader, val_loader, device)

    # test
    test_mae = evaluate_mae(model, test_loader, args.target, device)
    target_name, target_unit = QM9_TARGETS[args.target]
    print(f'\nTest MAE: {test_mae:.4f} [{target_unit}]  (target: {target_name})')

    # save csv (append to shared file; write header only if file is new)
    os.makedirs('results/csv', exist_ok=True)
    import csv
    csv_path = 'results/csv/qm9_results.csv'
    header = ['model', 'target', 'target_name', 'test_mae', 'unit', 'best_val_mae',
              'hid_dim', 'n_layers', 'epochs_trained', 'seed']
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([args.model, args.target, target_name, f'{test_mae:.6f}', target_unit,
                         f'{best_val:.6f}', args.hid_dim, args.n_layers, args.epochs, args.seed])
    print(f'Results appended to: {csv_path}')


if __name__ == '__main__':
    main()