#!/usr/bin/env python
"""Benchmark Stream 11 operations against NumPy."""
import numpy as np
import rumpy
import time


def timeit(func, n_runs=100):
    """Time a function over multiple runs."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return np.median(times) * 1000  # ms


def benchmark(name, rumpy_func, numpy_func, n_runs=100):
    """Benchmark rumpy vs numpy."""
    r_time = timeit(rumpy_func, n_runs)
    n_time = timeit(numpy_func, n_runs)
    ratio = r_time / n_time
    print(f"{name:30s} rumpy: {r_time:7.3f}ms  numpy: {n_time:7.3f}ms  ratio: {ratio:.2f}x")
    return ratio


def main():
    print("Stream 11 Performance Benchmark")
    print("=" * 70)
    print()

    # Setup arrays
    n_2d = np.random.rand(1000, 1000)
    r_2d = rumpy.asarray(n_2d)

    n_1d = np.random.rand(1000000)
    r_1d = rumpy.asarray(n_1d)

    # repeat
    print("REPEAT")
    benchmark("repeat 1D x3",
              lambda: rumpy.repeat(r_1d[:10000], 3),
              lambda: np.repeat(n_1d[:10000], 3))
    benchmark("repeat 2D axis=0",
              lambda: rumpy.repeat(r_2d[:100,:100], 3, axis=0),
              lambda: np.repeat(n_2d[:100,:100], 3, axis=0))
    print()

    # tile
    print("TILE")
    benchmark("tile 1D x10",
              lambda: rumpy.tile(r_1d[:1000], 10),
              lambda: np.tile(n_1d[:1000], 10))
    benchmark("tile 2D (3,3)",
              lambda: rumpy.tile(r_2d[:100,:100], (3, 3)),
              lambda: np.tile(n_2d[:100,:100], (3, 3)))
    print()

    # roll
    print("ROLL")
    benchmark("roll 1D flat",
              lambda: rumpy.roll(r_1d, 1000),
              lambda: np.roll(n_1d, 1000))
    benchmark("roll 2D axis=0",
              lambda: rumpy.roll(r_2d, 100, axis=0),
              lambda: np.roll(n_2d, 100, axis=0))
    print()

    # rot90
    print("ROT90")
    benchmark("rot90 k=1",
              lambda: rumpy.rot90(r_2d),
              lambda: np.rot90(n_2d))
    benchmark("rot90 k=3",
              lambda: rumpy.rot90(r_2d, k=3),
              lambda: np.rot90(n_2d, k=3))
    print()

    # pad
    print("PAD")
    benchmark("pad constant",
              lambda: rumpy.pad(r_2d[:100,:100], 10, mode='constant'),
              lambda: np.pad(n_2d[:100,:100], 10, mode='constant'))
    benchmark("pad edge",
              lambda: rumpy.pad(r_2d[:100,:100], 10, mode='edge'),
              lambda: np.pad(n_2d[:100,:100], 10, mode='edge'))
    print()

    # splits
    print("SPLITS")
    benchmark("hsplit 2D",
              lambda: rumpy.hsplit(r_2d, 10),
              lambda: np.hsplit(n_2d, 10))
    benchmark("vsplit 2D",
              lambda: rumpy.vsplit(r_2d, 10),
              lambda: np.vsplit(n_2d, 10))
    print()

    # stacks
    print("STACKS")
    n_arrs = [np.random.rand(100, 100) for _ in range(10)]
    r_arrs = [rumpy.asarray(a) for a in n_arrs]
    benchmark("column_stack",
              lambda: rumpy.column_stack(r_arrs),
              lambda: np.column_stack(n_arrs))
    benchmark("dstack",
              lambda: rumpy.dstack(r_arrs),
              lambda: np.dstack(n_arrs))
    print()

    print("=" * 70)
    print("Note: ratio < 1.0 means rumpy is faster")


if __name__ == "__main__":
    main()
