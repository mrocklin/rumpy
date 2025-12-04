"""Tests for views, slicing, reshape, and transpose."""

import numpy as np
import pytest

import rumpy as rp
from helpers import assert_eq


class TestSlicing:
    """Test basic slicing operations."""

    def test_simple_slice(self):
        r = rp.ones(10)
        n = np.ones(10)
        assert_eq(r[2:7], n[2:7])

    def test_slice_with_step(self):
        r = rp.ones(10)
        n = np.ones(10)
        assert_eq(r[::2], n[::2])

    def test_slice_negative_indices(self):
        r = rp.ones(10)
        n = np.ones(10)
        assert_eq(r[-5:], n[-5:])

    def test_slice_negative_step(self):
        r = rp.ones(10)
        n = np.ones(10)
        assert_eq(r[::-1], n[::-1])

    def test_slice_2d_single_axis(self):
        r = rp.ones((4, 5))
        n = np.ones((4, 5))
        assert_eq(r[1:3], n[1:3])

    def test_slice_2d_both_axes(self):
        r = rp.ones((4, 5))
        n = np.ones((4, 5))
        assert_eq(r[1:3, 2:4], n[1:3, 2:4])

    def test_slice_preserves_view(self):
        """Slicing should create a view, not a copy."""
        r = rp.ones(10)
        v = r[2:7]
        # Shape should be updated
        assert v.shape == (5,)
        # Strides should be same as original (no copy)
        assert v.strides == r.strides


class TestReshape:
    """Test reshape operations."""

    def test_reshape_1d_to_2d(self):
        r = rp.ones(12)
        n = np.ones(12)
        assert_eq(r.reshape(3, 4), n.reshape(3, 4))

    def test_reshape_2d_to_1d(self):
        r = rp.ones((3, 4))
        n = np.ones((3, 4))
        assert_eq(r.reshape(12), n.reshape(12))

    def test_reshape_2d_to_3d(self):
        r = rp.ones((6, 4))
        n = np.ones((6, 4))
        assert_eq(r.reshape(2, 3, 4), n.reshape(2, 3, 4))

    def test_reshape_invalid_size(self):
        r = rp.ones(10)
        with pytest.raises(ValueError):
            r.reshape(3, 4)  # 12 != 10


class TestTranspose:
    """Test transpose operations."""

    def test_transpose_2d(self):
        r = rp.ones((3, 4))
        n = np.ones((3, 4))
        assert_eq(r.transpose(), n.transpose())

    def test_transpose_2d_shape(self):
        r = rp.ones((3, 4))
        assert r.transpose().shape == (4, 3)

    def test_transpose_2d_strides(self):
        r = rp.ones((3, 4))
        # Original: C-order strides (32, 8)
        # Transposed: (8, 32)
        assert r.T.strides == (8, 32)

    def test_transpose_T_property(self):
        r = rp.ones((3, 4))
        n = np.ones((3, 4))
        assert_eq(r.T, n.T)

    def test_transpose_3d(self):
        r = rp.ones((2, 3, 4))
        n = np.ones((2, 3, 4))
        assert_eq(r.transpose(), n.transpose())


class TestViewProperties:
    """Test that views work correctly."""

    def test_slice_nbytes(self):
        """Sliced view reports correct nbytes."""
        r = rp.ones(10)
        v = r[2:7]
        assert v.nbytes == 5 * 8  # 5 float64s

    def test_reshape_is_view(self):
        """Reshape of contiguous array is a view."""
        r = rp.ones(12)
        r2 = r.reshape(3, 4)
        # Both point to same underlying data (can't test directly,
        # but strides should be recomputed for contiguous layout)
        assert r2.strides == (32, 8)
