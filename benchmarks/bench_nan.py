"""NaN-aware reduction benchmarks.

Each benchmark is a function that takes (xp, size) where xp is numpy or rumpy,
and returns a callable that performs the benchmark.
"""

import numpy as np


def nansum_sparse(xp, size):
    """Sum with ~10% NaN values."""
    data = np.random.randn(size)
    nan_idx = np.random.choice(size, size // 10, replace=False)
    data[nan_idx] = np.nan
    x = xp.array(data)

    def fn():
        return xp.nansum(x)
    return fn


def nanmean_sparse(xp, size):
    """Mean with ~10% NaN values."""
    data = np.random.randn(size)
    nan_idx = np.random.choice(size, size // 10, replace=False)
    data[nan_idx] = np.nan
    x = xp.array(data)

    def fn():
        return xp.nanmean(x)
    return fn


def nanstd_sparse(xp, size):
    """Std with ~10% NaN values."""
    data = np.random.randn(size)
    nan_idx = np.random.choice(size, size // 10, replace=False)
    data[nan_idx] = np.nan
    x = xp.array(data)

    def fn():
        return xp.nanstd(x)
    return fn


def nanmin_sparse(xp, size):
    """Min with ~10% NaN values."""
    data = np.random.randn(size)
    nan_idx = np.random.choice(size, size // 10, replace=False)
    data[nan_idx] = np.nan
    x = xp.array(data)

    def fn():
        return xp.nanmin(x)
    return fn


def nanmax_sparse(xp, size):
    """Max with ~10% NaN values."""
    data = np.random.randn(size)
    nan_idx = np.random.choice(size, size // 10, replace=False)
    data[nan_idx] = np.nan
    x = xp.array(data)

    def fn():
        return xp.nanmax(x)
    return fn


def nansum_dense(xp, size):
    """Sum with ~50% NaN values (worst case)."""
    data = np.random.randn(size)
    nan_idx = np.random.choice(size, size // 2, replace=False)
    data[nan_idx] = np.nan
    x = xp.array(data)

    def fn():
        return xp.nansum(x)
    return fn


def nanmean_2d_axis(xp, size):
    """Mean along axis with NaN values."""
    n = int(np.sqrt(size))
    data = np.random.randn(n, n)
    # Scatter NaNs
    for i in range(n):
        nan_cols = np.random.choice(n, n // 10, replace=False)
        data[i, nan_cols] = np.nan
    x = xp.array(data)

    def fn():
        return xp.nanmean(x, axis=1)
    return fn


def nanvar_2d_axis(xp, size):
    """Variance along axis with NaN values."""
    n = int(np.sqrt(size))
    data = np.random.randn(n, n)
    for i in range(n):
        nan_cols = np.random.choice(n, n // 10, replace=False)
        data[i, nan_cols] = np.nan
    x = xp.array(data)

    def fn():
        return xp.nanvar(x, axis=1)
    return fn


def nanargmin_1d(xp, size):
    """Argmin ignoring NaN."""
    data = np.random.randn(size)
    nan_idx = np.random.choice(size, size // 10, replace=False)
    data[nan_idx] = np.nan
    x = xp.array(data)

    def fn():
        return xp.nanargmin(x)
    return fn


def nanargmax_1d(xp, size):
    """Argmax ignoring NaN."""
    data = np.random.randn(size)
    nan_idx = np.random.choice(size, size // 10, replace=False)
    data[nan_idx] = np.nan
    x = xp.array(data)

    def fn():
        return xp.nanargmax(x)
    return fn


def clean_and_compute(xp, size):
    """Realistic workflow: compute stats on data with NaN."""
    data = np.random.randn(size)
    nan_idx = np.random.choice(size, size // 20, replace=False)
    data[nan_idx] = np.nan
    x = xp.array(data)

    def fn():
        mean = xp.nanmean(x)
        std = xp.nanstd(x)
        vmin = xp.nanmin(x)
        vmax = xp.nanmax(x)
        return mean + std + vmin + vmax  # combine results
    return fn


BENCHMARKS = [
    ("nansum (10% NaN)", nansum_sparse),
    ("nanmean (10% NaN)", nanmean_sparse),
    ("nanstd (10% NaN)", nanstd_sparse),
    ("nanmin (10% NaN)", nanmin_sparse),
    ("nanmax (10% NaN)", nanmax_sparse),
    ("nansum (50% NaN)", nansum_dense),
    ("nanmean 2D axis", nanmean_2d_axis),
    ("nanvar 2D axis", nanvar_2d_axis),
    ("nanargmin", nanargmin_1d),
    ("nanargmax", nanargmax_1d),
    ("clean and compute", clean_and_compute),
]
