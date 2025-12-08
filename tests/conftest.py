"""Pytest configuration and fixtures for rumpy tests.

See designs/testing.md for testing philosophy.
"""

import pytest

# === Dtype Tiers ===
# Tier 1: Representative (for simple wrappers)
FLOAT_DTYPES = ["float32", "float64"]
INT_DTYPES = ["int32", "int64"]
CORE_DTYPES = ["float64", "float32", "int64", "bool"]

# Tier 2: Full numeric (for operations with custom logic)
UINT_DTYPES = ["uint32", "uint64"]
NUMERIC_DTYPES = FLOAT_DTYPES + INT_DTYPES + UINT_DTYPES

# Tier 3: Complete (for dtype promotion tests)
COMPLEX_DTYPES = ["complex64", "complex128"]
ALL_DTYPES = NUMERIC_DTYPES + COMPLEX_DTYPES + ["bool"]

# Specialized
BITWISE_DTYPES = INT_DTYPES + UINT_DTYPES + ["bool"]

# === Shape Strategy ===
# Core shapes (always test)
CORE_SHAPES = [(5,), (3, 4), (2, 3, 4)]

# Edge case shapes
SHAPES_EMPTY = [(0,), (0, 5), (5, 0)]
SHAPES_SINGLE = [(1,), (1, 1)]

# Broadcasting pairs (shape_a, shape_b)
SHAPES_BROADCAST = [((3, 1), (1, 4)), ((1,), (5,)), ((2, 1, 4), (3, 1))]

# === Fixtures ===


@pytest.fixture(params=CORE_DTYPES, ids=lambda d: f"dtype={d}")
def dtype(request):
    """Core dtypes for general tests."""
    return request.param


@pytest.fixture(params=FLOAT_DTYPES, ids=lambda d: f"dtype={d}")
def float_dtype(request):
    """Float dtypes only."""
    return request.param


@pytest.fixture(params=INT_DTYPES, ids=lambda d: f"dtype={d}")
def int_dtype(request):
    """Signed integer dtypes."""
    return request.param


@pytest.fixture(params=NUMERIC_DTYPES, ids=lambda d: f"dtype={d}")
def numeric_dtype(request):
    """All numeric dtypes (floats + ints + uints)."""
    return request.param


@pytest.fixture(params=BITWISE_DTYPES, ids=lambda d: f"dtype={d}")
def bitwise_dtype(request):
    """Dtypes supporting bitwise operations."""
    return request.param


@pytest.fixture(params=CORE_SHAPES, ids=lambda s: f"shape={s}")
def shape(request):
    """Core shapes for general tests."""
    return request.param


@pytest.fixture(params=SHAPES_EMPTY, ids=lambda s: f"shape={s}")
def empty_shape(request):
    """Empty (zero-size) shapes."""
    return request.param
