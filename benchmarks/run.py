"""Benchmark runner for rumpy vs numpy."""

import time
import argparse
import importlib
from pathlib import Path

import numpy as np
import rumpy as rp


def time_fn(fn, warmup=1, repeats=5):
    """Time a function, return min time in seconds."""
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return min(times)


def run_benchmark(name, bench_fn, size, skip_reason=None):
    """Run a single benchmark for both numpy and rumpy."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")
    print(f"{'='*60}")

    if skip_reason:
        print(f"SKIPPED: {skip_reason}")
        return None, None

    print(f"{'Size':>12} {'NumPy (ms)':>12} {'Rumpy (ms)':>12} {'Speedup':>10}")
    print("-" * 50)

    results = []
    for sz in size:
        np_time = rp_time = None
        np_err = rp_err = None

        # NumPy
        try:
            np_fn = bench_fn(np, sz)
            np_time = time_fn(np_fn) * 1000
        except Exception as e:
            np_err = str(e)

        # Rumpy
        try:
            rp_fn = bench_fn(rp, sz)
            rp_time = time_fn(rp_fn) * 1000
        except Exception as e:
            rp_err = str(e)

        # Report
        if np_err:
            print(f"{sz:>12} NumPy ERROR: {np_err}")
        elif rp_err:
            print(f"{sz:>12} {np_time:>12.3f} RUMPY ERROR: {rp_err}")
        else:
            speedup = np_time / rp_time if rp_time > 0 else float('inf')
            print(f"{sz:>12} {np_time:>12.3f} {rp_time:>12.3f} {speedup:>9.2f}x")

        results.append((sz, np_time, rp_time, np_err, rp_err))

    return results


def discover_benchmarks():
    """Find all benchmark modules."""
    bench_dir = Path(__file__).parent
    modules = []
    for f in sorted(bench_dir.glob("bench_*.py")):
        module_name = f.stem
        modules.append(module_name)
    return modules


def load_benchmarks(module_name):
    """Load benchmarks from a module. Returns list of (name, fn, skip_reason)."""
    module = importlib.import_module(module_name)
    if hasattr(module, 'BENCHMARKS'):
        return module.BENCHMARKS
    return []


def main():
    parser = argparse.ArgumentParser(description="Run rumpy vs numpy benchmarks")
    parser.add_argument('--filter', '-f', type=str, help="Filter benchmarks by name")
    parser.add_argument('--sizes', '-s', type=str, default="1000,10000,100000",
                        help="Comma-separated sizes to test")
    parser.add_argument('--module', '-m', type=str, help="Run only specific module (e.g., bench_ufuncs)")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(',')]

    print("Rumpy vs NumPy Benchmarks")
    print("=" * 60)

    modules = discover_benchmarks()
    if args.module:
        modules = [m for m in modules if args.module in m]

    for module_name in modules:
        print(f"\n\n>>> Loading {module_name}")
        try:
            benchmarks = load_benchmarks(module_name)
            for item in benchmarks:
                if len(item) == 2:
                    name, bench_fn = item
                    skip_reason = None
                else:
                    name, bench_fn, skip_reason = item

                if args.filter and args.filter.lower() not in name.lower():
                    continue
                run_benchmark(name, bench_fn, sizes, skip_reason)
        except Exception as e:
            print(f"  Error loading module: {e}")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
