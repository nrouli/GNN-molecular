"""
Inference Time Benchmark Utility
=================================
Measures inference latency correctly following best practices:

1. Data pre-loaded on device (no host-device transfer in the loop)
2. GPU warm-up passes before measurement
3. CUDA Events for GPU timing (not CPU wall-clock)
4. torch.cuda.synchronize() before reading elapsed time
5. Multiple repetitions with statistical reporting (mean, std, median, percentiles)
6. Supports both GPU and CPU, and arbitrary input types (tensors, PyG Data, dicts, etc.)

Usage:
    from benchmark import benchmark_inference

    # For standard models (tensor input)
    results = benchmark_inference(model, dummy_input, device='cuda')

    # For PyG models (Data object input)
    from torch_geometric.data import Data
    data = Data(x=..., edge_index=..., pos=..., batch=...)
    results = benchmark_inference(model, data, device='cuda')

    # Compare two models at matched inference time
    compare_models(model_a, input_a, model_b, input_b, device='cuda')
"""

import time
import numpy as np
import torch


def _move_to_device(input_data, device):
    """Recursively move input data to device."""
    if isinstance(input_data, torch.Tensor):
        return input_data.to(device)
    elif isinstance(input_data, dict):
        return {k: _move_to_device(v, device) for k, v in input_data.items()}
    elif isinstance(input_data, (list, tuple)):
        moved = [_move_to_device(x, device) for x in input_data]
        return type(input_data)(moved)
    elif hasattr(input_data, 'to'):  # PyG Data, Batch, etc.
        return input_data.to(device)
    else:
        return input_data


def _run_forward(model, input_data):
    """Run a single forward pass, handling different input types."""
    if isinstance(input_data, dict):
        return model(**input_data)
    elif isinstance(input_data, (list, tuple)):
        return model(*input_data)
    else:
        return model(input_data)


def benchmark_inference(
    model,
    input_data,
    device='cuda',
    warmup_runs=50,
    repetitions=300,
    verbose=True,
):
    """
    Benchmark inference time of a model.

    Args:
        model: PyTorch model (nn.Module)
        input_data: Model input. Can be:
            - torch.Tensor
            - dict of tensors (for **kwargs forward)
            - list/tuple of tensors (for *args forward)
            - PyG Data/Batch object
        device: 'cuda' or 'cpu'
        warmup_runs: Number of warmup forward passes
        repetitions: Number of timed forward passes
        verbose: Print results

    Returns:
        dict with timing statistics in milliseconds
    """
    device = torch.device(device)
    use_cuda = device.type == 'cuda'

    if use_cuda and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device('cpu')
        use_cuda = False

    # Move model and data to device
    model = model.to(device)
    model.eval()
    input_data = _move_to_device(input_data, device)

    timings = np.zeros(repetitions)

    # ── GPU path: use CUDA Events ──
    if use_cuda:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        # Warmup: trigger lazy initialization, kernel loading, memory allocation
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = _run_forward(model, input_data)

        # Timed runs
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = _run_forward(model, input_data)
                ender.record()
                # Wait for GPU to finish
                torch.cuda.synchronize()
                timings[rep] = starter.elapsed_time(ender)  # milliseconds

    # ── CPU path: use time.perf_counter ──
    else:
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = _run_forward(model, input_data)

        # Timed runs
        with torch.no_grad():
            for rep in range(repetitions):
                start = time.perf_counter()
                _ = _run_forward(model, input_data)
                end = time.perf_counter()
                timings[rep] = (end - start) * 1000  # convert to ms

    # ── Statistics ──
    results = {
        'mean_ms': np.mean(timings),
        'std_ms': np.std(timings),
        'median_ms': np.median(timings),
        'min_ms': np.min(timings),
        'max_ms': np.max(timings),
        'p95_ms': np.percentile(timings, 95),
        'p99_ms': np.percentile(timings, 99),
        'repetitions': repetitions,
        'device': str(device),
        'all_timings_ms': timings,
    }

    if verbose:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{'=' * 55}")
        print(f"  Inference Benchmark  ({device})")
        print(f"{'=' * 55}")
        print(f"  Model params:    {num_params:,}")
        print(f"  Warmup runs:     {warmup_runs}")
        print(f"  Timed runs:      {repetitions}")
        print(f"{'─' * 55}")
        print(f"  Mean:            {results['mean_ms']:.4f} ms")
        print(f"  Std:             {results['std_ms']:.4f} ms")
        print(f"  Median:          {results['median_ms']:.4f} ms")
        print(f"  Min:             {results['min_ms']:.4f} ms")
        print(f"  Max:             {results['max_ms']:.4f} ms")
        print(f"  95th percentile: {results['p95_ms']:.4f} ms")
        print(f"  99th percentile: {results['p99_ms']:.4f} ms")
        print(f"  Throughput:      {1000 / results['mean_ms']:.1f} samples/sec")
        print(f"{'=' * 55}\n")

    return results


def compare_models(
    model_a,
    input_a,
    model_b,
    input_b,
    device='cuda',
    warmup_runs=50,
    repetitions=300,
    name_a='Model A',
    name_b='Model B',
):
    """
    Compare inference time of two models side by side.

    Args:
        model_a, model_b: Two PyTorch models
        input_a, input_b: Corresponding inputs
        device: 'cuda' or 'cpu'
        warmup_runs, repetitions: Benchmark parameters
        name_a, name_b: Display names

    Returns:
        dict with results for both models and comparison stats
    """
    print(f"\nBenchmarking {name_a}...")
    res_a = benchmark_inference(
        model_a, input_a, device=device,
        warmup_runs=warmup_runs, repetitions=repetitions, verbose=False,
    )

    print(f"Benchmarking {name_b}...")
    res_b = benchmark_inference(
        model_b, input_b, device=device,
        warmup_runs=warmup_runs, repetitions=repetitions, verbose=False,
    )

    params_a = sum(p.numel() for p in model_a.parameters() if p.requires_grad)
    params_b = sum(p.numel() for p in model_b.parameters() if p.requires_grad)

    speedup = res_b['mean_ms'] / res_a['mean_ms']

    print(f"\n{'=' * 65}")
    print(f"  Model Comparison  ({device})")
    print(f"{'=' * 65}")
    print(f"  {'':30s} {name_a:>15s} {name_b:>15s}")
    print(f"{'─' * 65}")
    print(f"  {'Parameters':30s} {params_a:>15,} {params_b:>15,}")
    print(f"  {'Mean latency (ms)':30s} {res_a['mean_ms']:>15.4f} {res_b['mean_ms']:>15.4f}")
    print(f"  {'Std (ms)':30s} {res_a['std_ms']:>15.4f} {res_b['std_ms']:>15.4f}")
    print(f"  {'Median (ms)':30s} {res_a['median_ms']:>15.4f} {res_b['median_ms']:>15.4f}")
    print(f"  {'95th percentile (ms)':30s} {res_a['p95_ms']:>15.4f} {res_b['p95_ms']:>15.4f}")
    print(f"  {'Throughput (samples/sec)':30s} {1000/res_a['mean_ms']:>15.1f} {1000/res_b['mean_ms']:>15.1f}")
    print(f"{'─' * 65}")
    print(f"  {name_a} is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than {name_b}")
    print(f"{'=' * 65}\n")

    return {
        name_a: res_a,
        name_b: res_b,
        'speedup_a_over_b': speedup,
    }


# ── Quick self-test ─────────────────────────────────────────────────
if __name__ == '__main__':
    # Simple test with a dummy MLP
    class DummyModel(torch.nn.Module):
        def __init__(self, dim=256, layers=4):
            super().__init__()
            self.net = torch.nn.Sequential(*[
                torch.nn.Sequential(
                    torch.nn.Linear(dim, dim),
                    torch.nn.SiLU(),
                ) for _ in range(layers)
            ])

        def forward(self, x):
            return self.net(x)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DummyModel(dim=256, layers=4)
    dummy_input = torch.randn(32, 256)

    results = benchmark_inference(model, dummy_input, device=device, repetitions=100)
