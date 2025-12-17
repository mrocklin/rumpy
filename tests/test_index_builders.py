"""Tests for Stream 34: Advanced Index Builders

Functions: ix_, ogrid, mgrid, fill_diagonal, ndenumerate, ndindex
"""

import numpy as np
import pytest
import rumpy as rp

from helpers import assert_eq


class TestIx:
    """Tests for np.ix_ - open mesh from sequences."""

    def test_basic_2d(self):
        """Basic 2D open mesh."""
        x = np.array([1, 2, 3])
        y = np.array([4, 5])
        np_result = np.ix_(x, y)
        rp_result = rp.ix_(rp.asarray(x), rp.asarray(y))
        assert len(rp_result) == len(np_result)
        for r, n in zip(rp_result, np_result):
            assert_eq(r, n)

    def test_basic_3d(self):
        """3D open mesh."""
        x = np.array([1, 2])
        y = np.array([3, 4, 5])
        z = np.array([6, 7])
        np_result = np.ix_(x, y, z)
        rp_result = rp.ix_(rp.asarray(x), rp.asarray(y), rp.asarray(z))
        assert len(rp_result) == len(np_result)
        for r, n in zip(rp_result, np_result):
            assert_eq(r, n)

    def test_single_array(self):
        """Single array returns 1D."""
        x = np.array([1, 2, 3, 4])
        np_result = np.ix_(x)
        rp_result = rp.ix_(rp.asarray(x))
        assert len(rp_result) == 1
        assert_eq(rp_result[0], np_result[0])

    @pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
    def test_dtype_preservation(self, dtype):
        """Dtype should be preserved."""
        x = np.array([1, 2, 3], dtype=dtype)
        y = np.array([4, 5], dtype=dtype)
        np_result = np.ix_(x, y)
        rp_result = rp.ix_(rp.asarray(x), rp.asarray(y))
        for r, n in zip(rp_result, np_result):
            assert str(r.dtype) == str(n.dtype)


class TestOgrid:
    """Tests for np.ogrid - open sparse meshgrid."""

    def test_basic_2d(self):
        """Basic 2D sparse grid."""
        np_result = np.ogrid[0:3, 0:2]
        rp_result = rp.ogrid[0:3, 0:2]
        assert len(rp_result) == len(np_result)
        for r, n in zip(rp_result, np_result):
            assert_eq(r, n)

    def test_basic_1d(self):
        """1D range."""
        np_result = np.ogrid[0:5]
        rp_result = rp.ogrid[0:5]
        assert_eq(rp_result, np_result)

    def test_with_step(self):
        """Integer step."""
        np_result = np.ogrid[0:10:2]
        rp_result = rp.ogrid[0:10:2]
        assert_eq(rp_result, np_result)

    def test_complex_step(self):
        """Complex step means linspace-like (num points)."""
        np_result = np.ogrid[0:1:5j]
        rp_result = rp.ogrid[0:1:5j]
        assert_eq(rp_result, np_result)

    def test_3d(self):
        """3D sparse grid."""
        np_result = np.ogrid[0:2, 0:3, 0:4]
        rp_result = rp.ogrid[0:2, 0:3, 0:4]
        assert len(rp_result) == 3
        for r, n in zip(rp_result, np_result):
            assert_eq(r, n)

    def test_mixed_step(self):
        """Mixed integer and complex steps."""
        np_result = np.ogrid[0:4, 0:1:3j]
        rp_result = rp.ogrid[0:4, 0:1:3j]
        assert len(rp_result) == 2
        for r, n in zip(rp_result, np_result):
            assert_eq(r, n)


class TestMgrid:
    """Tests for np.mgrid - dense meshgrid."""

    def test_basic_2d(self):
        """Basic 2D dense grid."""
        np_result = np.mgrid[0:3, 0:2]
        rp_result = rp.mgrid[0:3, 0:2]
        assert len(rp_result) == len(np_result)
        for r, n in zip(rp_result, np_result):
            assert_eq(r, n)

    def test_basic_1d(self):
        """1D range."""
        np_result = np.mgrid[0:5]
        rp_result = rp.mgrid[0:5]
        assert_eq(rp_result, np_result)

    def test_with_step(self):
        """Integer step."""
        np_result = np.mgrid[0:10:2]
        rp_result = rp.mgrid[0:10:2]
        assert_eq(rp_result, np_result)

    def test_complex_step(self):
        """Complex step means linspace-like."""
        np_result = np.mgrid[0:1:5j]
        rp_result = rp.mgrid[0:1:5j]
        assert_eq(rp_result, np_result)

    def test_3d(self):
        """3D dense grid."""
        np_result = np.mgrid[0:2, 0:3, 0:4]
        rp_result = rp.mgrid[0:2, 0:3, 0:4]
        assert len(rp_result) == 3
        for r, n in zip(rp_result, np_result):
            assert_eq(r, n)


class TestFillDiagonal:
    """Tests for np.fill_diagonal - fill diagonal in-place."""

    def test_square_scalar(self):
        """Fill square matrix diagonal with scalar."""
        n = np.zeros((3, 3))
        r = rp.zeros((3, 3))
        np.fill_diagonal(n, 5)
        rp.fill_diagonal(r, 5)
        assert_eq(r, n)

    def test_rectangular_tall(self):
        """Fill tall rectangular matrix."""
        n = np.zeros((5, 3))
        r = rp.zeros((5, 3))
        np.fill_diagonal(n, 1)
        rp.fill_diagonal(r, 1)
        assert_eq(r, n)

    def test_rectangular_wide(self):
        """Fill wide rectangular matrix."""
        n = np.zeros((3, 5))
        r = rp.zeros((3, 5))
        np.fill_diagonal(n, 2)
        rp.fill_diagonal(r, 2)
        assert_eq(r, n)

    def test_array_values(self):
        """Fill with array of values."""
        n = np.zeros((3, 3))
        r = rp.zeros((3, 3))
        vals = [1, 2, 3]
        np.fill_diagonal(n, vals)
        rp.fill_diagonal(r, vals)
        assert_eq(r, n)

    def test_array_values_cyclic(self):
        """Array values cycle if shorter than diagonal."""
        n = np.zeros((5, 5))
        r = rp.zeros((5, 5))
        vals = [1, 2]
        np.fill_diagonal(n, vals)
        rp.fill_diagonal(r, vals)
        assert_eq(r, n)

    def test_wrap_true(self):
        """Wrap=True fills beyond first diagonal for tall matrices."""
        n = np.zeros((5, 3))
        r = rp.zeros((5, 3))
        np.fill_diagonal(n, 1, wrap=True)
        rp.fill_diagonal(r, 1, wrap=True)
        assert_eq(r, n)

    def test_wrap_false(self):
        """Wrap=False (default) stops at first diagonal."""
        n = np.zeros((5, 3))
        r = rp.zeros((5, 3))
        np.fill_diagonal(n, 1, wrap=False)
        rp.fill_diagonal(r, 1, wrap=False)
        assert_eq(r, n)

    def test_3d(self):
        """Fill 3D array diagonal."""
        n = np.zeros((3, 3, 3))
        r = rp.zeros((3, 3, 3))
        np.fill_diagonal(n, [1, 2, 3])
        rp.fill_diagonal(r, [1, 2, 3])
        assert_eq(r, n)

    @pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
    def test_dtypes(self, dtype):
        """Fill diagonal preserves array dtype."""
        n = np.zeros((4, 4), dtype=dtype)
        r = rp.zeros((4, 4), dtype=dtype)
        np.fill_diagonal(n, 7)
        rp.fill_diagonal(r, 7)
        assert_eq(r, n)


class TestNdenumerate:
    """Tests for np.ndenumerate - multi-dimensional index iterator."""

    def test_1d(self):
        """1D array enumeration."""
        n = np.array([1, 2, 3])
        r = rp.asarray(n)
        np_result = list(np.ndenumerate(n))
        rp_result = list(rp.ndenumerate(r))
        assert len(rp_result) == len(np_result)
        for (rp_idx, rp_val), (np_idx, np_val) in zip(rp_result, np_result):
            assert rp_idx == np_idx
            assert rp_val == np_val

    def test_2d(self):
        """2D array enumeration."""
        n = np.array([[1, 2], [3, 4]])
        r = rp.asarray(n)
        np_result = list(np.ndenumerate(n))
        rp_result = list(rp.ndenumerate(r))
        assert len(rp_result) == len(np_result)
        for (rp_idx, rp_val), (np_idx, np_val) in zip(rp_result, np_result):
            assert rp_idx == np_idx
            assert rp_val == np_val

    def test_3d(self):
        """3D array enumeration."""
        n = np.arange(24).reshape(2, 3, 4)
        r = rp.asarray(n)
        np_result = list(np.ndenumerate(n))
        rp_result = list(rp.ndenumerate(r))
        assert len(rp_result) == len(np_result)
        for (rp_idx, rp_val), (np_idx, np_val) in zip(rp_result, np_result):
            assert rp_idx == np_idx
            assert rp_val == np_val


class TestNdindex:
    """Tests for np.ndindex - N-dimensional index iterator."""

    def test_basic_2d(self):
        """Basic 2D index iteration."""
        np_result = list(np.ndindex(2, 3))
        rp_result = list(rp.ndindex(2, 3))
        assert rp_result == np_result

    def test_basic_3d(self):
        """Basic 3D index iteration."""
        np_result = list(np.ndindex(2, 2, 2))
        rp_result = list(rp.ndindex(2, 2, 2))
        assert rp_result == np_result

    def test_1d(self):
        """1D index iteration."""
        np_result = list(np.ndindex(5))
        rp_result = list(rp.ndindex(5))
        assert rp_result == np_result

    def test_from_tuple(self):
        """Index iteration from tuple shape."""
        shape = (2, 3, 4)
        np_result = list(np.ndindex(shape))
        rp_result = list(rp.ndindex(shape))
        assert rp_result == np_result

    def test_empty(self):
        """Empty shape yields single empty tuple."""
        np_result = list(np.ndindex())
        rp_result = list(rp.ndindex())
        assert rp_result == np_result

    def test_with_zero_dim(self):
        """Shape with zero dimension yields empty iteration."""
        np_result = list(np.ndindex(2, 0, 3))
        rp_result = list(rp.ndindex(2, 0, 3))
        assert rp_result == np_result
