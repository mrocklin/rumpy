"""Tests for broadcasting."""

import numpy as np
import pytest

import rumpy
from helpers import assert_eq


class TestBroadcastShapes:
    """Test shape broadcasting rules."""

    def test_same_shape(self):
        r1 = rumpy.arange(6).reshape([2, 3])
        r2 = rumpy.arange(6, 12).reshape([2, 3])
        n1 = np.arange(6, dtype=np.float64).reshape(2, 3)
        n2 = np.arange(6, 12, dtype=np.float64).reshape(2, 3)
        assert_eq(r1 + r2, n1 + n2)

    def test_broadcast_1d_to_2d(self):
        # (3,) + (2, 3) -> (2, 3)
        r1 = rumpy.arange(3)
        r2 = rumpy.arange(6).reshape([2, 3])
        n1 = np.arange(3, dtype=np.float64)
        n2 = np.arange(6, dtype=np.float64).reshape(2, 3)
        assert_eq(r1 + r2, n1 + n2)

    def test_broadcast_column_to_2d(self):
        # (2, 1) + (2, 3) -> (2, 3)
        r1 = rumpy.arange(2).reshape([2, 1])
        r2 = rumpy.arange(6).reshape([2, 3])
        n1 = np.arange(2, dtype=np.float64).reshape(2, 1)
        n2 = np.arange(6, dtype=np.float64).reshape(2, 3)
        assert_eq(r1 + r2, n1 + n2)

    def test_broadcast_row_column(self):
        # (1, 3) + (2, 1) -> (2, 3)
        r1 = rumpy.arange(3).reshape([1, 3])
        r2 = rumpy.arange(2).reshape([2, 1])
        n1 = np.arange(3, dtype=np.float64).reshape(1, 3)
        n2 = np.arange(2, dtype=np.float64).reshape(2, 1)
        assert_eq(r1 + r2, n1 + n2)

    def test_broadcast_3d(self):
        # (3,) + (2, 3, 4) -> (2, 3, 4) with broadcast along axis 2
        r1 = rumpy.arange(4)
        r2 = rumpy.arange(24).reshape([2, 3, 4])
        n1 = np.arange(4, dtype=np.float64)
        n2 = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        assert_eq(r1 + r2, n1 + n2)

    def test_broadcast_mul(self):
        r1 = rumpy.arange(3)
        r2 = rumpy.arange(6).reshape([2, 3])
        n1 = np.arange(3, dtype=np.float64)
        n2 = np.arange(6, dtype=np.float64).reshape(2, 3)
        assert_eq(r1 * r2, n1 * n2)

    def test_broadcast_sub(self):
        r1 = rumpy.arange(3).reshape([1, 3])
        r2 = rumpy.arange(2).reshape([2, 1])
        n1 = np.arange(3, dtype=np.float64).reshape(1, 3)
        n2 = np.arange(2, dtype=np.float64).reshape(2, 1)
        assert_eq(r1 - r2, n1 - n2)

    def test_broadcast_div(self):
        r1 = rumpy.arange(1, 4).reshape([1, 3])
        r2 = rumpy.arange(1, 3).reshape([2, 1])
        n1 = np.arange(1, 4, dtype=np.float64).reshape(1, 3)
        n2 = np.arange(1, 3, dtype=np.float64).reshape(2, 1)
        assert_eq(r1 / r2, n1 / n2)


class TestBroadcastErrors:
    """Test that incompatible shapes raise errors."""

    def test_incompatible_shapes(self):
        r1 = rumpy.arange(3)
        r2 = rumpy.arange(4)
        with pytest.raises(ValueError):
            r1 + r2

    def test_incompatible_2d(self):
        r1 = rumpy.arange(6).reshape([2, 3])
        r2 = rumpy.arange(8).reshape([2, 4])
        with pytest.raises(ValueError):
            r1 + r2


class TestBroadcastWithViews:
    """Test broadcasting works with sliced views."""

    def test_broadcast_sliced_1d(self):
        r_full = rumpy.arange(10)
        r1 = r_full[2:5]  # shape (3,)
        r2 = rumpy.arange(6).reshape([2, 3])
        n_full = np.arange(10, dtype=np.float64)
        n1 = n_full[2:5]
        n2 = np.arange(6, dtype=np.float64).reshape(2, 3)
        assert_eq(r1 + r2, n1 + n2)

    def test_broadcast_transposed(self):
        r1 = rumpy.arange(6).reshape([2, 3])
        r2 = rumpy.arange(6).reshape([3, 2]).T  # (2, 3) via transpose
        n1 = np.arange(6, dtype=np.float64).reshape(2, 3)
        n2 = np.arange(6, dtype=np.float64).reshape(3, 2).T
        assert_eq(r1 + r2, n1 + n2)


class TestOuterProduct:
    """Test outer product style broadcasting."""

    def test_outer_add(self):
        # Outer add: (3,) + (4, 1) -> (4, 3)
        r1 = rumpy.arange(3)
        r2 = rumpy.arange(4).reshape([4, 1])
        n1 = np.arange(3, dtype=np.float64)
        n2 = np.arange(4, dtype=np.float64).reshape(4, 1)
        assert_eq(r1 + r2, n1 + n2)

    def test_outer_mul(self):
        # Outer product: (3, 1) * (1, 4) -> (3, 4)
        r1 = rumpy.arange(3).reshape([3, 1])
        r2 = rumpy.arange(4).reshape([1, 4])
        n1 = np.arange(3, dtype=np.float64).reshape(3, 1)
        n2 = np.arange(4, dtype=np.float64).reshape(1, 4)
        assert_eq(r1 * r2, n1 * n2)
