"""Array method benchmarks (Stream 18).

Each benchmark is a function that takes (xp, size) where xp is numpy or rumpy,
and returns a callable that performs the benchmark.
"""

import numpy as np


def nonzero_sparse(xp, size):
    """Find nonzero indices in sparse array (33% nonzero)."""
    data = np.arange(size) % 3
    x = xp.array(data)

    def fn():
        return x.nonzero()
    return fn


def nonzero_dense(xp, size):
    """Find nonzero indices in dense array (90% nonzero)."""
    data = np.random.randn(size)
    data[::10] = 0  # 10% zeros
    x = xp.array(data)

    def fn():
        return x.nonzero()
    return fn


def argsort_random(xp, size):
    """Argsort random data."""
    data = np.random.randn(size)
    x = xp.array(data)

    def fn():
        return x.argsort()
    return fn


def argsort_reversed(xp, size):
    """Argsort reversed (worst case for some algorithms)."""
    data = np.arange(size)[::-1].astype(float)
    x = xp.array(data.copy())

    def fn():
        return x.argsort()
    return fn


def sort_inplace_random(xp, size):
    """In-place sort of random data."""
    data = np.random.randn(size)

    def fn():
        x = xp.array(data.copy())
        x.sort()
        return x
    return fn


def sort_inplace_reversed(xp, size):
    """In-place sort of reversed data."""
    data = np.arange(size)[::-1].astype(float)

    def fn():
        x = xp.array(data.copy())
        x.sort()
        return x
    return fn


def searchsorted_random(xp, size):
    """Search for random values in sorted array."""
    sorted_data = np.arange(size, dtype=float)
    x = xp.array(sorted_data)
    # Search for size/10 random values
    search_vals = np.random.rand(size // 10) * size
    v = xp.array(search_vals)

    def fn():
        return x.searchsorted(v)
    return fn


def repeat_flat(xp, size):
    """Repeat elements (flattened)."""
    data = np.arange(size // 10, dtype=float)
    x = xp.array(data)

    def fn():
        return x.repeat(10)
    return fn


def repeat_axis0(xp, size):
    """Repeat rows along axis 0."""
    n = int(np.sqrt(size))
    data = np.random.randn(n, n)
    x = xp.array(data)

    def fn():
        return x.repeat(2, axis=0)
    return fn


def take_method_flat(xp, size):
    """Array.take() method (flattened)."""
    data = np.random.randn(size)
    x = xp.array(data)
    idx = xp.array(np.random.randint(0, size, size // 10))

    def fn():
        return x.take(idx)
    return fn


def take_method_axis0(xp, size):
    """Array.take() method along axis 0."""
    n = int(np.sqrt(size))
    data = np.random.randn(n, n)
    x = xp.array(data)
    idx = xp.array(np.random.randint(0, n, n // 4))

    def fn():
        return x.take(idx, axis=0)
    return fn


def fill_float64(xp, size):
    """Fill array with scalar value."""
    def fn():
        x = xp.zeros(size)
        x.fill(42.0)
        return x
    return fn


def tobytes_contiguous(xp, size):
    """Convert array to bytes."""
    data = np.arange(size, dtype=float)
    x = xp.array(data)

    def fn():
        return x.tobytes()
    return fn


def partition_random(xp, size):
    """Partition random data at median."""
    data = np.random.randn(size)

    def fn():
        x = xp.array(data.copy())
        x.partition(size // 2)
        return x
    return fn


def partition_reversed(xp, size):
    """Partition reversed data at median."""
    data = np.arange(size, dtype=float)[::-1]

    def fn():
        x = xp.array(data.copy())
        x.partition(size // 2)
        return x
    return fn


def argpartition_random(xp, size):
    """Argpartition random data at median."""
    data = np.random.randn(size)
    x = xp.array(data)

    def fn():
        return x.argpartition(size // 2)
    return fn


BENCHMARKS = [
    ("nonzero (sparse)", nonzero_sparse),
    ("nonzero (dense)", nonzero_dense),
    ("argsort (random)", argsort_random),
    ("argsort (reversed)", argsort_reversed),
    ("sort in-place (random)", sort_inplace_random),
    ("sort in-place (reversed)", sort_inplace_reversed),
    ("searchsorted", searchsorted_random),
    ("repeat (flat)", repeat_flat),
    ("repeat (axis 0)", repeat_axis0),
    ("take method (flat)", take_method_flat),
    ("take method (axis 0)", take_method_axis0),
    ("fill", fill_float64),
    ("tobytes", tobytes_contiguous),
    ("partition (random)", partition_random),
    ("partition (reversed)", partition_reversed),
    ("argpartition", argpartition_random),
]
