"""
Verification & Benchmark: Original vs Matrix-Isomorphism CliffordAlgebra
=========================================================================
Run from your project root:
    python verify_algebra.py

Tests:
  1. Round-trip: multivector -> matrix -> multivector (identity check)
  2. Geometric product: matrix path vs Cayley path (numerical agreement)
  3. b() / q() / norm(): new vs old (numerical agreement)
  4. Timing: old geometric_product vs new, across batch sizes
"""

import torch
import time
import sys

# ── Adjust these imports to match your project structure ──
# Old algebra
from algebra.cliffordalgebra import CliffordAlgebra as OldAlgebra

# New algebra (matrix isomorphism)
from gpu_algebra.cliffordalgebra import CliffordAlgebra as NewAlgebra


from gpu_algebra.matrix_kernel import (
    get_matrix_mapping,
    ga_to_matrix,
    matrix_to_ga,
    complex_matmul_broadcast,
)


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_round_trip(signature, device):
    """Test: multivector -> matrix -> multivector should be identity."""
    separator("Round-Trip Test")

    mapping = get_matrix_mapping(signature, device, torch.float32)

    # Single multivector
    a = torch.randn(8, device=device)
    a_rt = matrix_to_ga(ga_to_matrix(a, mapping), mapping)
    err = (a - a_rt).abs().max().item()
    print(f"  Single MV max error:  {err:.2e}  {'PASS' if err < 1e-5 else 'FAIL'}")

    # Batched
    a_batch = torch.randn(64, 8, device=device)
    a_batch_rt = matrix_to_ga(ga_to_matrix(a_batch, mapping), mapping)
    err = (a_batch - a_batch_rt).abs().max().item()
    print(f"  Batched MV max error: {err:.2e}  {'PASS' if err < 1e-5 else 'FAIL'}")


def test_geometric_product(metric, device):
    """Test: matrix-iso GP matches Cayley-table GP."""
    separator("Geometric Product Agreement")

    old = OldAlgebra(metric).to(device)
    new = NewAlgebra(metric).to(device)

    # Random multivectors
    for shape_desc, shape in [("single", (8,)), ("batch", (64, 8)), ("2D batch", (16, 32, 8))]:
        a = torch.randn(*shape, device=device)
        b = torch.randn(*shape, device=device)

        gp_old = old.geometric_product(a, b)
        gp_new = new.geometric_product(a, b)

        err = (gp_old - gp_new).abs().max().item()
        print(f"  {shape_desc:>10s} {str(shape):>15s}  max error: {err:.2e}  {'PASS' if err < 1e-4 else 'FAIL'}")


def test_derived_ops(metric, device):
    """Test: b(), q(), norm() agreement between old and new."""
    separator("Derived Operations (b, q, norm)")

    old = OldAlgebra(metric).to(device)
    new = NewAlgebra(metric).to(device)

    a = torch.randn(64, 8, device=device)

    # q (quadratic form)
    q_old = old.q(a)
    q_new = new.q(a)
    err = (q_old - q_new).abs().max().item()
    print(f"  q()    max error: {err:.2e}  {'PASS' if err < 1e-4 else 'FAIL'}")

    # norm
    n_old = old.norm(a)
    n_new = new.norm(a)
    err = (n_old - n_new).abs().max().item()
    print(f"  norm() max error: {err:.2e}  {'PASS' if err < 1e-4 else 'FAIL'}")

    # norms (per-grade)
    norms_old = old.norms(a)
    norms_new = new.norms(a)
    for i, (no, nn) in enumerate(zip(norms_old, norms_new)):
        err = (no - nn).abs().max().item()
        print(f"  norms()[{i}] max error: {err:.2e}  {'PASS' if err < 1e-4 else 'FAIL'}")


def benchmark_gp(metric, device, warmup=50, reps=300):
    """Benchmark: geometric_product old vs new."""
    separator("Geometric Product Benchmark")

    old = OldAlgebra(metric).to(device)
    new = NewAlgebra(metric).to(device)

    for batch_size in [64, 256, 1024, 4096]:
        a = torch.randn(batch_size, 8, device=device)
        b = torch.randn(batch_size, 8, device=device)

        # ── Old ──
        with torch.no_grad():
            for _ in range(warmup):
                _ = old.geometric_product(a, b)
            if device.type == 'cuda':
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(reps):
                _ = old.geometric_product(a, b)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            dt_old = (time.perf_counter() - t0) / reps * 1000

        # ── New ──
        with torch.no_grad():
            for _ in range(warmup):
                _ = new.geometric_product(a, b)
            if device.type == 'cuda':
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(reps):
                _ = new.geometric_product(a, b)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            dt_new = (time.perf_counter() - t0) / reps * 1000

        speedup = dt_old / dt_new if dt_new > 0 else float('inf')
        print(f"  batch={batch_size:>5d}  old: {dt_old:.4f} ms  new: {dt_new:.4f} ms  speedup: {speedup:.2f}x")


def benchmark_norm(metric, device, warmup=50, reps=300):
    """Benchmark: norm() old vs new."""
    separator("Norm Benchmark")

    old = OldAlgebra(metric).to(device)
    new = NewAlgebra(metric).to(device)

    a = torch.randn(1024, 8, device=device)

    # ── Old ──
    with torch.no_grad():
        for _ in range(warmup):
            _ = old.norm(a)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(reps):
            _ = old.norm(a)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        dt_old = (time.perf_counter() - t0) / reps * 1000

    # ── New ──
    with torch.no_grad():
        for _ in range(warmup):
            _ = new.norm(a)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(reps):
            _ = new.norm(a)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        dt_new = (time.perf_counter() - t0) / reps * 1000

    speedup = dt_old / dt_new if dt_new > 0 else float('inf')
    print(f"  norm(1024x8)  old: {dt_old:.4f} ms  new: {dt_new:.4f} ms  speedup: {speedup:.2f}x")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    metric = (1.0, 1.0, 1.0)
    # Correctness
    test_round_trip(metric, device)
    test_geometric_product(metric, device)
    test_derived_ops(metric, device)

    # Performance
    benchmark_gp(metric, device)
    benchmark_norm(metric, device)

    separator("Done")


if __name__ == "__main__":
    main()
