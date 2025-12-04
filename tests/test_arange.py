"""Tests for arange and element access."""

import numpy as np
import pytest

import rumpy as rp
from helpers import assert_eq


class TestArange:
    """Test rp.arange against np.arange."""

    def test_simple(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert_eq(r, n)

    def test_start_stop(self):
        r = rp.arange(2, 10)
        n = np.arange(2, 10, dtype=np.float64)
        assert_eq(r, n)

    def test_start_stop_step(self):
        r = rp.arange(0, 10, 2)
        n = np.arange(0, 10, 2, dtype=np.float64)
        assert_eq(r, n)

    def test_float_step(self):
        r = rp.arange(0, 1, 0.1)
        n = np.arange(0, 1, 0.1, dtype=np.float64)
        assert_eq(r, n)

    def test_negative_step(self):
        r = rp.arange(10, 0, -1)
        n = np.arange(10, 0, -1, dtype=np.float64)
        assert_eq(r, n)

    def test_dtype_int64(self):
        r = rp.arange(10, dtype="int64")
        n = np.arange(10, dtype=np.int64)
        assert_eq(r, n)

    def test_dtype_float32(self):
        r = rp.arange(10, dtype="float32")
        n = np.arange(10, dtype=np.float32)
        assert_eq(r, n)


class TestIntegerIndexing:
    """Test integer indexing for element access."""

    def test_1d_single_index(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        # Integer index returns array with shape [1]
        assert_eq(r[5], n[5:6])

    def test_1d_negative_index(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert_eq(r[-1], n[-1:])

    def test_2d_full_index(self):
        r = rp.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        # Full integer index returns scalar
        assert r[1, 2] == n[1, 2]

    def test_2d_negative_indices(self):
        r = rp.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert r[-1, -1] == n[-1, -1]

    def test_2d_mixed_int_slice(self):
        r = rp.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        # Mixed: integer reduces dim, slice keeps it
        r_row = r[1, :]
        n_row = n[1:2, :]
        assert_eq(r_row, n_row)


class TestSlicingWithArange:
    """Test slicing with arange-generated arrays."""

    def test_slice_values(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert_eq(r[2:7], n[2:7])

    def test_slice_step(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert_eq(r[::2], n[::2])

    def test_slice_reverse(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert_eq(r[::-1], n[::-1])
