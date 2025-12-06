"""Statistical computation benchmarks.

Each benchmark is a function that takes (xp, size) where xp is numpy or rumpy,
and returns a callable that performs the benchmark.
"""

import numpy as np


def zscore_2d(xp, size):
    """Z-score normalization on 2D array along axis."""
    n = int(np.sqrt(size))
    data = np.random.randn(n, n)
    x = xp.array(data)

    def fn():
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        return (x - mean) / std
    return fn


def weighted_mean(xp, size):
    """Weighted mean: sum(x * w) / sum(w)"""
    x = xp.array(np.random.randn(size))
    w = xp.array(np.random.rand(size))

    def fn():
        return (x * w).sum() / w.sum()
    return fn


def covariance_manual(xp, size):
    """Manual covariance: E[(X-mu_x)(Y-mu_y)]"""
    x = xp.array(np.random.randn(size))
    y = xp.array(np.random.randn(size))

    def fn():
        x_mean = x.mean()
        y_mean = y.mean()
        return ((x - x_mean) * (y - y_mean)).mean()
    return fn


def variance_axis(xp, size):
    """Variance along axis for a 2D array."""
    n = int(np.sqrt(size))
    x = xp.array(np.random.randn(n, n))

    def fn():
        return x.var(axis=0)
    return fn


def cumsum(xp, size):
    """Cumulative sum."""
    x = xp.array(np.random.randn(size))

    def fn():
        return xp.cumsum(x)
    return fn


def diff_cumsum_chain(xp, size):
    """Compute diff then cumsum (tests chaining)."""
    data = np.cumsum(np.random.randn(size))  # smooth signal
    x = xp.array(data)

    def fn():
        d = xp.diff(x)
        return xp.cumsum(d)
    return fn


def histogram_bincount(xp, size):
    """Histogram via bincount (if available)."""
    # Generate integers in range [0, 100)
    data = np.random.randint(0, 100, size)
    x = xp.array(data)

    def fn():
        # This requires xp.bincount
        return xp.bincount(x)
    return fn


def percentile(xp, size):
    """Compute percentiles."""
    x = xp.array(np.random.randn(size))

    def fn():
        return xp.percentile(x, [25, 50, 75])
    return fn


BENCHMARKS = [
    ("z-score 2D (axis)", zscore_2d),
    ("weighted mean", weighted_mean),
    ("covariance (manual)", covariance_manual),
    ("variance along axis", variance_axis),
    ("cumsum", cumsum),  # requires xp.cumsum module-level
    ("diff + cumsum chain", diff_cumsum_chain),  # requires xp.diff, xp.cumsum
    ("histogram (bincount)", histogram_bincount),  # requires xp.bincount
    ("percentile", percentile),  # requires xp.percentile
]
