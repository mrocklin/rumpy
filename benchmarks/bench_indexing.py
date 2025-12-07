"""Indexing operation benchmarks.

Each benchmark is a function that takes (xp, size) where xp is numpy or rumpy,
and returns a callable that performs the benchmark.
"""

import numpy as np


def take_random(xp, size):
    """Take random elements by index."""
    data = np.random.randn(size)
    x = xp.array(data)
    # Take 10% of elements randomly
    idx = xp.array(np.random.randint(0, size, size // 10))

    def fn():
        return xp.take(x, idx)
    return fn


def take_sequential(xp, size):
    """Take sequential chunk by index."""
    data = np.random.randn(size)
    x = xp.array(data)
    # Take a contiguous block
    idx = xp.array(np.arange(size // 4, size // 2))

    def fn():
        return xp.take(x, idx)
    return fn


def take_2d_axis0(xp, size):
    """Take rows from 2D array."""
    n = int(np.sqrt(size))
    data = np.random.randn(n, n)
    x = xp.array(data)
    idx = xp.array(np.random.randint(0, n, n // 4))

    def fn():
        return xp.take(x, idx, axis=0)
    return fn


def take_2d_axis1(xp, size):
    """Take columns from 2D array."""
    n = int(np.sqrt(size))
    data = np.random.randn(n, n)
    x = xp.array(data)
    idx = xp.array(np.random.randint(0, n, n // 4))

    def fn():
        return xp.take(x, idx, axis=1)
    return fn


def compress_sparse(xp, size):
    """Compress with ~10% True values."""
    data = np.random.randn(size)
    x = xp.array(data)
    cond = xp.array(np.random.rand(size) < 0.1)

    def fn():
        return xp.compress(cond, x)
    return fn


def compress_dense(xp, size):
    """Compress with ~50% True values."""
    data = np.random.randn(size)
    x = xp.array(data)
    cond = xp.array(np.random.rand(size) < 0.5)

    def fn():
        return xp.compress(cond, x)
    return fn


def searchsorted_random(xp, size):
    """Search for random values in sorted array."""
    sorted_data = np.sort(np.random.randn(size))
    x = xp.array(sorted_data)
    # Search for 1000 values
    values = xp.array(np.random.randn(1000))

    def fn():
        return xp.searchsorted(x, values)
    return fn


def searchsorted_sequential(xp, size):
    """Search for sequential values (best case for binary search)."""
    sorted_data = np.sort(np.random.randn(size))
    x = xp.array(sorted_data)
    # Search for values that span the range evenly
    values = xp.array(np.linspace(sorted_data.min(), sorted_data.max(), 1000))

    def fn():
        return xp.searchsorted(x, values)
    return fn


def argwhere_sparse(xp, size):
    """Find indices where condition is true (~10%)."""
    data = np.random.randn(size)
    x = xp.array(data > 1.28)  # ~10% of normal distribution

    def fn():
        return xp.argwhere(x)
    return fn


def argwhere_dense(xp, size):
    """Find indices where condition is true (~50%)."""
    data = np.random.randn(size)
    x = xp.array(data > 0)  # ~50%

    def fn():
        return xp.argwhere(x)
    return fn


def put_random(xp, size):
    """Put values at random indices."""
    data = np.random.randn(size).copy()
    x = xp.array(data)
    idx = xp.array(np.random.randint(0, size, size // 10))
    values = xp.array(np.zeros(size // 10))

    def fn():
        # Note: put modifies in place, need fresh array each time
        y = x.copy()
        xp.put(y, idx, values)
        return y
    return fn


def choose_3way(xp, size):
    """Choose from 3 arrays based on index."""
    choices = [
        xp.array(np.random.randn(size)),
        xp.array(np.random.randn(size)),
        xp.array(np.random.randn(size)),
    ]
    idx = xp.array(np.random.randint(0, 3, size))

    def fn():
        return xp.choose(idx, choices)
    return fn


def threshold_select(xp, size):
    """Realistic: select values above threshold."""
    data = np.random.randn(size)
    x = xp.array(data)
    threshold = 1.0

    def fn():
        mask = x > threshold
        return xp.compress(mask, x)
    return fn


def binned_lookup(xp, size):
    """Realistic: bin data and lookup bin edges."""
    data = np.random.randn(size)
    x = xp.array(data)
    bins = xp.array(np.linspace(-3, 3, 100))

    def fn():
        return xp.searchsorted(bins, x)
    return fn


BENCHMARKS = [
    ("take random", take_random),
    ("take sequential", take_sequential),
    ("take 2D axis=0", take_2d_axis0),
    ("take 2D axis=1", take_2d_axis1),
    ("compress sparse (10%)", compress_sparse),
    ("compress dense (50%)", compress_dense),
    ("searchsorted random", searchsorted_random),
    ("searchsorted sequential", searchsorted_sequential),
    ("argwhere sparse", argwhere_sparse),
    ("argwhere dense", argwhere_dense),
    ("put random", put_random),
    ("choose 3-way", choose_3way),
    ("threshold select", threshold_select),
    ("binned lookup", binned_lookup),
]
