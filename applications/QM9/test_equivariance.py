"""
Equivariance tests for GA-GAT on scalar molecular property prediction.

Since the model outputs a scalar per molecule, the relevant property is
INVARIANCE of the output under each symmetry group:

    f(g · x) = f(x)   for all g in G

We test four groups:
    SO(3)  — proper rotations
    O(3)   — rotations + reflections  (improper rotations)
    SE(3)  — rotations + translations
    E(3)   — rotations + reflections + translations

For each, we:
  1. Build a small random molecular graph
  2. Forward pass on the original geometry
  3. Transform positions, forward pass again
  4. Compare outputs (absolute difference vs tolerance)
"""

import torch
import numpy as np
from torch_geometric.data import Data
from scipy.spatial.transform import Rotation as R

# ── Adjust this import to match your project layout ──────────────────────────
from models.GAGAT import GA_GAT

# ─── Helpers ─────────────────────────────────────────────────────────────────

def random_rotation_matrix(device="cpu", dtype=torch.float64):
    """Sample a uniformly random SO(3) rotation matrix."""
    rot = R.random().as_matrix()
    return torch.tensor(rot, device=device, dtype=dtype)


def random_reflection_matrix(device="cpu", dtype=torch.float64):
    """Sample a random improper rotation (rotation * parity flip)."""
    rot = random_rotation_matrix(device, dtype)
    # Flip the first axis to get det = -1
    rot[:, 0] *= -1
    return rot


def random_translation(dim=3, scale=10.0, device="cpu", dtype=torch.float64):
    """Sample a random translation vector."""
    return torch.randn(dim, device=device, dtype=dtype) * scale


def build_toy_molecule(n_atoms=12, n_species=5, cutoff=4.0,
                       device="cpu", dtype=torch.float64):
    """
    Create a small random molecular graph with:
      - random atom types z in [1, n_species]
      - random 3D positions
      - edges between all pairs within `cutoff`
      - single-molecule batch index
    """
    z = torch.randint(1, n_species + 1, (n_atoms,), device=device)
    pos = torch.randn(n_atoms, 3, device=device, dtype=dtype) * 2.0

    # Build edges via distance cutoff (no self-loops)
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)          # (N, N, 3)
    dists = diff.norm(dim=-1)                            # (N, N)
    mask = (dists < cutoff) & (dists > 0)
    src, dst = mask.nonzero(as_tuple=True)
    edge_index = torch.stack([src, dst], dim=0)

    batch = torch.zeros(n_atoms, dtype=torch.long, device=device)

    data = Data(z=z, pos=pos, edge_index=edge_index, batch=batch)
    return data


def transform_data(data, rotation=None, translation=None):
    """
    Apply an affine transformation to a Data object's positions.
    Returns a new Data with transformed pos (everything else shared).
    """
    pos = data.pos.clone()
    if rotation is not None:
        pos = pos @ rotation.T          # (N, 3) @ (3, 3)
    if translation is not None:
        pos = pos + translation.unsqueeze(0)
    return Data(z=data.z, pos=pos, edge_index=data.edge_index, batch=data.batch)


# ─── Test runner ─────────────────────────────────────────────────────────────

def test_invariance(model, data, rotation=None, translation=None,
                    label="", atol=1e-4):
    """
    Forward pass on original and transformed data.
    Reports absolute error and pass/fail.
    """
    with torch.no_grad():
        y_orig = model(data)
        data_t = transform_data(data, rotation=rotation, translation=translation)
        y_trans = model(data_t)

    err = (y_orig - y_trans).abs().max().item()
    status = "PASS" if err < atol else "FAIL"
    print(f"  [{status}]  {label:20s}  |  max |Δy| = {err:.2e}  (tol {atol:.0e})")
    return err, status


def run_all_tests(n_trials=5, device="cpu", dtype=torch.float64, atol=1e-4):
    """
    Instantiate the model once, then run multiple random trials
    for each symmetry group.
    """
    print("=" * 70)
    print("GA-GAT  Equivariance / Invariance Tests")
    print("=" * 70)
    print(f"  device : {device}")
    print(f"  dtype  : {dtype}")
    print(f"  trials : {n_trials}")
    print(f"  atol   : {atol:.0e}")
    print()

    # ── Model ────────────────────────────────────────────────────────────
    model = GA_GAT(
        max_z=10,
        hidden_dim=32,
        out_dim=1,
        num_layers=2,
        num_heads=4,
        num_rbf=16,
        cutoff=5.0,
        dropout=0.0,
        embed_positions=True,
        readout="mean",
    )
    model = model.to(device=device, dtype=dtype)
    model.eval()

    # Embedding layer stays int; cast only float params
    # (nn.Embedding ignores .to(dtype) for its weight, which is fine;
    #  but the lookup output will be float32.  We cast it below.)

    results = {}

    for trial in range(n_trials):
        print(f"── Trial {trial + 1}/{n_trials} "
              + "─" * 50)
        data = build_toy_molecule(n_atoms=10, n_species=5, cutoff=5.0,
                                  device=device, dtype=dtype)

        # SO(3): pure rotation
        rot = random_rotation_matrix(device, dtype)
        err, st = test_invariance(model, data, rotation=rot,
                                  label="SO(3)  rotation", atol=atol)
        results.setdefault("SO(3)", []).append((err, st))

        # O(3): improper rotation (det = -1)
        irot = random_reflection_matrix(device, dtype)
        err, st = test_invariance(model, data, rotation=irot,
                                  label="O(3)   reflection", atol=atol)
        results.setdefault("O(3)", []).append((err, st))

        # SE(3): rotation + translation
        rot2 = random_rotation_matrix(device, dtype)
        t = random_translation(device=device, dtype=dtype)
        err, st = test_invariance(model, data, rotation=rot2, translation=t,
                                  label="SE(3)  rot+trans", atol=atol)
        results.setdefault("SE(3)", []).append((err, st))

        # E(3): improper rotation + translation
        irot2 = random_reflection_matrix(device, dtype)
        t2 = random_translation(device=device, dtype=dtype)
        err, st = test_invariance(model, data, rotation=irot2, translation=t2,
                                  label="E(3)   full", atol=atol)
        results.setdefault("E(3)", []).append((err, st))

        print()

    # ── Summary ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    for group in ["SO(3)", "O(3)", "SE(3)", "E(3)"]:
        errs = [e for e, _ in results[group]]
        passes = sum(1 for _, s in results[group] if s == "PASS")
        max_err = max(errs)
        mean_err = np.mean(errs)
        tag = "ALL PASS" if passes == n_trials else f"{passes}/{n_trials} PASS"
        print(f"  {group:6s}  {tag:12s}  "
              f"mean |Δy| = {mean_err:.2e}   max |Δy| = {max_err:.2e}")
    print()


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # float64 gives a cleaner invariance signal; float32 accumulates
    # more numerical noise, so raise atol accordingly.
    run_all_tests(
        n_trials=5,
        device=device,
        dtype=torch.float64,
        atol=1e-4,
    )

    # Optional: repeat in float32 with a looser tolerance
    print("\n" + "#" * 70)
    print("# Re-running in float32 (looser tolerance)")
    print("#" * 70 + "\n")
    run_all_tests(
        n_trials=5,
        device=device,
        dtype=torch.float32,
        atol=5e-3,
    )
