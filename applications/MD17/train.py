"""
Training script for energy + force prediction on (revised) MD17.

Usage:
    python datasets/MD17/train.py --model schnet --molecule "aspirin"
    python datasets/MD17/train.py --model egnn --molecule "ethanol" --epochs 750 --hid_dim 128
    python datasets/MD17/train.py --model schnet --train_all_molecules
    python datasets/MD17/train.py --model schnet --scheduler cosine_wr --T_0 50 --T_mult 2

Run from the project root (GNN-molecular/).

Notes on the training objective:
    MD17/rMD17 provide both potential energy E and atomic forces F.
    The standard objective is a weighted sum of energy and force MAE,
    with forces obtained by autograd of E w.r.t. atomic positions:
        F_pred = - dE_pred / dr
    Training on energy alone is underdetermined and typically diverges
    or plateaus on these small datasets (1k train samples per molecule).
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
    set_seed, get_device, build_cgenn, build_egnn, build_ga_gat, build_gat, count_parameters,
    get_md17_data, save_checkpoint,
    RMD17_MOLECULES, SEED,
)
from utils.metrics_tracker import MetricsTracker
from utils.util import SchNetWrapper


def parse_args():
    parser = argparse.ArgumentParser('MD17 Training (energy + forces)')

    # model
    parser.add_argument('--model', default='schnet',
                        choices=['egnn', 'gat', 'schnet', 'cgenn'])
    parser.add_argument('--hid_dim',  type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_heads',  type=int, default=4)

    # data
    parser.add_argument('--molecule',   type=str, default='aspirin',
                        help='MD17 molecule name (e.g. "aspirin")')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--split_idx',  type=int, default=1, help='Train/val split index from dataset (values from 1 to 5)')
    parser.add_argument('--cutoff',     type=float, default=5.0,
                        help='Radius graph cutoff in Angstrom')

    # loss weights
    parser.add_argument('--energy_coeff', type=float, default=0.05,
                        help='Energy loss weight (rMD17 convention: 0.05)')
    parser.add_argument('--force_coeff',  type=float, default=0.95,
                        help='Force loss weight (rMD17 convention: 0.95)')

    # training
    parser.add_argument('--epochs',       type=int,   default=1000)
    parser.add_argument('--lr',           type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-8)
    parser.add_argument('--patience',     type=int,   default=1000,
                        help='Early stopping patience (on val force MAE)')
    parser.add_argument('--step',         type=int,   default=1,
                        help='Log every N epochs')
    parser.add_argument('--seed',         type=int,   default=SEED)
    parser.add_argument('--grad_clip',    type=float, default=10.0,
                        help='Gradient norm clip (0 to disable)')

    # scheduler
    parser.add_argument('--scheduler', default='cosine_wr',
                        choices=['plateau', 'cosine', 'cosine_wr', 'none'])
    # plateau
    parser.add_argument('--lr_patience',  type=int,   default=50)
    parser.add_argument('--lr_factor',    type=float, default=0.5)
    parser.add_argument('--min_lr',       type=float, default=1e-6)
    # cosine / cosine_wr
    parser.add_argument('--T_0',    type=int, default=50,
                        help='cosine_wr: epochs until first restart')
    parser.add_argument('--T_mult', type=int, default=2,
                        help='cosine_wr: factor to grow cycle length after each restart')
    parser.add_argument('--eta_min', type=float, default=1e-6,
                        help='cosine / cosine_wr: minimum LR')

    # sweep
    parser.add_argument('--train_all_molecules', action='store_true',
                        help='Train on all revised MD17 molecules sequentially')

    # mixed precision 
    parser.add_argument('--mixed_precision', action='store_true', help='Enable or disable mixed precision arithmetic (if enabled bfloat16 will be used)')
    
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Forward pass: returns predicted energy AND enables autograd-based forces.
# ---------------------------------------------------------------------------
def forward_energy(model, data):
    """
    Run the model, returning a scalar energy per graph.

    IMPORTANT: data.pos must already have requires_grad_() set by the caller
    so that autograd can compute forces = -dE/dr afterwards. We do NOT clone
    positions here; the model must use the SAME tensor we differentiate wrt.
    """
    out = model(data)
    # normalize shape to [num_graphs]
    if out.dim() == 2 and out.size(-1) == 1:
        out = out.squeeze(-1)
    return out


def compute_energy_and_forces(model, data):
    """
    Predict energy and get forces = -dE/dpos via autograd.
    Assumes data.pos has requires_grad_() set.
    """
    energy = forward_energy(model, data)
    # grad of sum(E) wrt pos gives per-atom -F, summed across graphs (which
    # is what we want because each atom's force only depends on its own graph's energy)
    grad_outputs = torch.ones_like(energy)
    forces = -torch.autograd.grad(
        outputs=energy,
        inputs=data.pos,
        grad_outputs=grad_outputs,
        create_graph=model.training,   # needed for 2nd-order backprop through force loss
        retain_graph=True,
    )[0]
    return energy, forces


def compute_md17_stats(train_loader, device):
    energies = []
    num_atoms_ref = None
    for data in train_loader:
        if num_atoms_ref is None:
            num_atoms_ref = int((data.batch == 0).sum().item())
        energies.append(data.energy.view(-1).float())
    energies = torch.cat(energies)
    energy_mean = energies.mean().to(device)
  
    force_rms = torch.tensor(1.0, device=device)
    return energy_mean, force_rms, num_atoms_ref


def denormalize_energy(energy_hat, force_rms, energy_mean, num_atoms):
    return energy_hat + energy_mean


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def train_one_epoch(
    model, loader, optimizer, criterion, device,
    energy_atom_mean, force_rms, num_atoms,
    energy_coeff, force_coeff, grad_clip,
    scheduler=None, scheduler_is_batchwise=False, epoch=0,
):
    args = parse_args()
    model.train()
    loss_acc = 0.0
    e_mae_acc = 0.0
    f_mae_acc = 0.0
    n_iters = len(loader)

    bar = tqdm(loader, desc='[Train]', leave=False)
    for step, data in enumerate(bar):
        data = data.to(device)
        # CRITICAL: enable grad on positions so we can get forces by autograd
        data.pos.requires_grad_(True)

        optimizer.zero_grad()    
        energy_hat, forces = compute_energy_and_forces(model, data)

        # denormalize energy back to physical units for loss against raw target
        energy_pred = denormalize_energy(energy_hat, force_rms, energy_atom_mean, num_atoms)

        energy_true = data.energy.view(-1).float()
        force_true = data.force.float()

        loss_e = criterion(energy_pred, energy_true)
        loss_f = criterion(forces, force_true)
        loss = energy_coeff * loss_e + force_coeff * loss_f

        
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # per-batch scheduler step (cosine / cosine_wr)
        if scheduler is not None and scheduler_is_batchwise:
            scheduler.step(epoch + step / n_iters)

        loss_acc += loss.item()
        e_mae_acc += loss_e.item()
        f_mae_acc += loss_f.item()
        bar.set_postfix(loss=f'{loss.item():.4f}',
                        e=f'{loss_e.item():.4f}',
                        f=f'{loss_f.item():.4f}')

    return loss_acc / n_iters, e_mae_acc / n_iters, f_mae_acc / n_iters


@torch.enable_grad()  # forces need grad even in eval
def evaluate(model, loader, criterion, device,
             energy_atom_mean, force_rms, num_atoms):
    model.eval()
    e_mae_acc = 0.0
    f_mae_acc = 0.0
    n = 0

    for data in loader:
        data = data.to(device)
        data.pos.requires_grad_(True)

        energy_hat, forces = compute_energy_and_forces(model, data)
        energy_pred = denormalize_energy(energy_hat, force_rms, energy_atom_mean, num_atoms)

        energy_true = data.energy.view(-1).float()
        force_true = data.force.float()

        bs = data.num_graphs
        e_mae_acc += (energy_pred - energy_true).abs().mean().item() * bs
        f_mae_acc += (forces - force_true).abs().mean().item() * bs
        n += bs

    return e_mae_acc / n, f_mae_acc / n


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------
def build_scheduler(args, optimizer):
    """
    Returns (scheduler, mode) where mode is one of:
        'plateau'   : step(val_metric) at end of each epoch
        'epoch'     : step() at end of each epoch
        'batch'     : step(epoch + i/iters) inside the training loop per batch
        'none'      : do nothing
    """
    name = args.scheduler
    if name == 'none':
        return None, 'none'

    if name == 'plateau':
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_factor,
            patience=args.lr_patience, min_lr=args.min_lr,
        )
        return sch, 'plateau'

    if name == 'cosine':
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.eta_min,
        )
        return sch, 'epoch'

    if name == 'cosine_wr':
        # T_0 in EPOCH units; we step per batch with fractional epoch
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min,
        )
        return sch, 'batch'

    raise ValueError(f'Unknown scheduler: {name}')


# ---------------------------------------------------------------------------
# Glue
# ---------------------------------------------------------------------------
def molecule_tag(molecule):
    return molecule.replace(' ', '_').replace('revised_', 'r')


def train_model(args, model, train_loader, val_loader, device, mol_name):
    tag = molecule_tag(mol_name)
    run_name = f'{args.model}_MD17_{tag}'

    # stats from the training set only
    energy_atom_mean, force_rms, num_atoms = compute_md17_stats(train_loader, device)
    print(f'Stats: E_atom_mean={energy_atom_mean.item():.4f}, '
          f'F_rms={force_rms.item():.4f}, N_atoms={num_atoms}')

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler, sch_mode = build_scheduler(args, optimizer)
    criterion = nn.L1Loss()

    metrics = MetricsTracker(run_name)

    best_val_f_mae = float('inf')
    best_state = None
    best_epoch = 0
    no_improve = 0

    print(f'\n{"="*60}')
    print(f'Training: {run_name}')
    print(f'Molecule: {mol_name}')
    print(f'Parameters: {count_parameters(model):,}')
    print(f'Scheduler: {args.scheduler}  (mode={sch_mode})')
    print(f'Loss weights: E={args.energy_coeff}, F={args.force_coeff}')
    print(f'{"="*60}\n')

    total_start = time.time()

    for epoch in range(args.epochs):
        t0 = time.time()

        train_loss, train_e_mae, train_f_mae = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            energy_atom_mean, force_rms, num_atoms,
            args.energy_coeff, args.force_coeff, args.grad_clip,
            scheduler=scheduler,
            scheduler_is_batchwise=(sch_mode == 'batch'),
            epoch=epoch,
        )
        val_e_mae, val_f_mae = evaluate(
            model, val_loader, criterion, device,
            energy_atom_mean, force_rms, num_atoms,
        )

        # per-epoch scheduler step
        if sch_mode == 'plateau':
            scheduler.step(val_f_mae)
        elif sch_mode == 'epoch':
            scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        dt = time.time() - t0

        if epoch % args.step == 0:
            metrics.update(epoch, train_f_mae, val_f_mae)  # track force MAE as primary

        if val_f_mae < best_val_f_mae:
            best_val_f_mae = val_f_mae
            best_epoch = epoch
            best_state = {
                'epoch': epoch,
                'state_dict': copy.deepcopy(model.state_dict()),
                'optimizer': copy.deepcopy(optimizer.state_dict()),
                'val_e_mae': val_e_mae,
                'val_f_mae': val_f_mae,
            }
            no_improve = 0
        else:
            no_improve += 1

        print(
            f'Epoch {epoch:>4} | '
            f'Train E/F: {train_e_mae:.4f}/{train_f_mae:.4f} | '
            f'Val E/F: {val_e_mae:.4f}/{val_f_mae:.4f} | '
            f'LR: {lr:.2e} | Best F: {best_val_f_mae:.4f} | {dt:.1f}s'
        )

        if no_improve >= args.patience:
            print(f'Early stopping at epoch {epoch} (patience={args.patience})')
            break

    total_time = time.time() - total_start
    print(f'\nTraining complete: {total_time/60:.1f} min')
    print(f'Best val force MAE: {best_val_f_mae:.6f} [kcal/mol/A] at epoch {best_epoch}')

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
            'val_f_mae': best_val_f_mae,
            'val_e_mae': best_state['val_e_mae'] if best_state else None,
            'energy_atom_mean': energy_atom_mean.item(),
            'force_rms': force_rms.item(),
            'num_atoms': num_atoms,
            'seed': args.seed,
        },
        save_dir='pretrained_models',
    )

    metrics.save()
    return metrics, best_val_f_mae, best_state, (energy_atom_mean, force_rms, num_atoms)


def run_single_molecule(args, device):
    train_loader, val_loader, test_loader, _stats_unused, in_dim = get_md17_data(
        molecule=args.molecule, batch_size=args.batch_size, split_idx=args.split_idx,
        cutoff=args.cutoff, seed=args.seed,
    )

    energies = []
    forces = []
    for data in train_loader:
        energies.append(data.energy.view(-1).float())
        forces.append(data.force.float())
    energies = torch.cat(energies)
    forces = torch.cat(forces, dim=0)
    n_atoms = int((next(iter(train_loader)).batch == 0).sum().item())

    print(f'N atoms per molecule: {n_atoms}')
    print(f'Energy: mean={energies.mean():.4f}, std={energies.std():.4f}, '
        f'min={energies.min():.4f}, max={energies.max():.4f}')
    print(f'Per-atom energy mean: {(energies / n_atoms).mean():.4f}')
    print(f'Force: mean={forces.mean():.4f}, std={forces.std():.4f}, '
        f'rms={forces.pow(2).mean().sqrt():.4f}, '
        f'abs_mean={forces.abs().mean():.4f}')

    match args.model:
        case 'egnn':
            model = build_egnn(in_dim=in_dim, hid_dim=args.hid_dim,
                               out_dim=1, n_layers=args.n_layers)
        case 'gat':
            model = build_gat(in_dim=in_dim, hid_dim=args.hid_dim,
                              out_dim=1, n_layers=args.n_layers,
                              n_heads=args.n_heads)
        case 'schnet':
            model = SchNetWrapper(
                hidden_channels=args.hid_dim,
                num_filters=args.hid_dim,
                num_interactions=args.n_layers,
                num_gaussians=25,
                cutoff=args.cutoff,
            )
        case 'cgenn':
            model = build_cgenn(in_dim=in_dim, hid_dim=args.hid_dim,
                                out_dim=1, n_layers=args.n_layers)
        case _:
            raise ValueError(f'Unknown model: {args.model}')

    model = model.to(device)
    metrics, best_val_f, best_state, (e_mean, f_rms, n_atoms) = train_model(
        args, model, train_loader, val_loader, device, args.molecule,
    )

    criterion = nn.L1Loss()
    test_e_mae, test_f_mae = evaluate(
        model, test_loader, criterion, device, e_mean, f_rms, n_atoms,
    )
    print(f'\nTest E MAE: {test_e_mae:.6f} [kcal/mol]')
    print(f'Test F MAE: {test_f_mae:.6f} [kcal/mol/A]   (molecule: {args.molecule})')

    return test_e_mae, test_f_mae, best_val_f


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    print(f'\n--- MD17 Energy + Force Prediction ---')

    if args.train_all_molecules:
        results = []
        for mol in RMD17_MOLECULES:
            print(f'\n{"#"*60}\n# Molecule: {mol}\n{"#"*60}')
            args.molecule = mol
            test_e, test_f, best_val_f = run_single_molecule(args, device)
            results.append({
                'molecule': mol,
                'test_energy_mae': test_e,
                'test_force_mae': test_f,
                'best_val_force_mae': best_val_f,
            })

        print(f'\n{"="*72}')
        print(f'{"Molecule":<28} {"Test E MAE":>14} {"Test F MAE":>14} {"Val F MAE":>12}')
        print(f'{"-"*72}')
        for r in results:
            print(f'{r["molecule"]:<28} '
                  f'{r["test_energy_mae"]:>14.6f} '
                  f'{r["test_force_mae"]:>14.6f} '
                  f'{r["best_val_force_mae"]:>12.6f}')
        print(f'{"="*72}')

        os.makedirs('results/csv', exist_ok=True)
        csv_path = f'results/csv/md17_{args.model}_all_molecules.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['molecule', 'test_energy_mae', 'test_force_mae', 'best_val_force_mae'],
            )
            writer.writeheader()
            writer.writerows(results)
        print(f'Results saved: {csv_path}')

    else:
        test_e, test_f, best_val_f = run_single_molecule(args, device)

        os.makedirs('results/csv', exist_ok=True)
        tag = molecule_tag(args.molecule)
        csv_path = f'results/csv/md17_{args.model}_{tag}.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['model', 'molecule',
                             'test_energy_mae', 'test_force_mae', 'best_val_force_mae',
                             'hid_dim', 'n_layers', 'scheduler', 'seed'])
            writer.writerow([args.model, args.molecule,
                             f'{test_e:.6f}', f'{test_f:.6f}', f'{best_val_f:.6f}',
                             args.hid_dim, args.n_layers,
                             args.scheduler, args.seed])
        print(f'Results saved: {csv_path}')


import numpy as np

def autocorr_fft(x, max_lag=None):
    """Normalized autocorrelation of a 1D time series via FFT.
    x: (T,) array. Returns c(0..max_lag-1) with c(0) = 1."""
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    T = len(x)
    if max_lag is None:
        max_lag = T // 4
    n = 2 ** int(np.ceil(np.log2(2 * T)))
    f = np.fft.fft(x, n=n)
    acf = np.fft.ifft(f * np.conj(f)).real[:T]
    acf /= np.arange(T, 0, -1)            # unbiased: divide by (T - tau)
    return acf[:max_lag] / acf[0]

def autocorr_batch(X, max_lag=None):
    """Same thing but for (T, K) matrix — runs K parallel series at once."""
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    T, K = X.shape
    if max_lag is None:
        max_lag = T // 4
    n = 2 ** int(np.ceil(np.log2(2 * T)))
    F = np.fft.fft(X, n=n, axis=0)
    acf = np.fft.ifft(F * np.conj(F), axis=0).real[:T]
    acf /= np.arange(T, 0, -1)[:, None]
    acf /= acf[0:1, :]
    return acf[:max_lag]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.load('data/MD17/rmd17/rmd17_aspirin.npz')



if __name__ == '__main__':
    main()
    
    