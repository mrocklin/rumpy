"""Tests for memory layout utilities (Stream 36).

Tests compare rumpy against numpy for:
- ascontiguousarray, asfortranarray
- require
- copyto
- broadcast_shapes
- concat, permute_dims, matrix_transpose (NumPy 2.0 aliases)
"""

import numpy as np
import pytest

import rumpy as rp
from conftest import CORE_DTYPES, FLOAT_DTYPES
from helpers import assert_eq


class TestAscontiguousarray:
    """Tests for ascontiguousarray."""

    @pytest.mark.parametrize("dtype", CORE_DTYPES)
    def test_already_contiguous(self, dtype):
        """Already C-contiguous array should return same or copy."""
        n = np.array([1, 2, 3], dtype=dtype)
        r = rp.asarray(n)
        result = rp.ascontiguousarray(r)
        assert_eq(result, np.ascontiguousarray(n))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_noncontiguous_slice(self, dtype):
        """Non-contiguous slice should become contiguous."""
        n = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
        r = rp.asarray(n)
        # Slice to make non-contiguous
        n_slice = n[:, ::2]
        r_slice = r[:, ::2]
        result = rp.ascontiguousarray(r_slice)
        expected = np.ascontiguousarray(n_slice)
        assert_eq(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtype_conversion(self, dtype):
        """Test dtype parameter."""
        n = np.array([1.5, 2.5, 3.5], dtype="float64")
        r = rp.asarray(n)
        result = rp.ascontiguousarray(r, dtype=dtype)
        expected = np.ascontiguousarray(n, dtype=dtype)
        assert_eq(result, expected)
        assert str(result.dtype) == dtype

    def test_preserves_dtype(self):
        """Should preserve input dtype by default."""
        for dtype in FLOAT_DTYPES:
            n = np.array([1.0, 2.0], dtype=dtype)
            r = rp.asarray(n)
            result = rp.ascontiguousarray(r)
            assert str(result.dtype) == dtype


class TestAsfortranarray:
    """Tests for asfortranarray."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_c_to_f_order(self, dtype):
        """C-order array converted to F-order."""
        n = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
        r = rp.asarray(n)
        result = rp.asfortranarray(r)
        expected = np.asfortranarray(n)
        assert_eq(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtype_conversion(self, dtype):
        """Test dtype parameter."""
        n = np.array([[1, 2], [3, 4]], dtype="float64")
        r = rp.asarray(n)
        result = rp.asfortranarray(r, dtype=dtype)
        expected = np.asfortranarray(n, dtype=dtype)
        assert_eq(result, expected)
        assert str(result.dtype) == dtype


class TestRequire:
    """Tests for require function."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_c_contiguous(self, dtype):
        """Test C_CONTIGUOUS requirement."""
        n = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
        r = rp.asarray(n)
        result = rp.require(r, requirements="C")
        expected = np.require(n, requirements="C")
        assert_eq(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_with_dtype(self, dtype):
        """Test dtype parameter."""
        n = np.array([1, 2, 3], dtype="float64")
        r = rp.asarray(n)
        result = rp.require(r, dtype=dtype)
        expected = np.require(n, dtype=dtype)
        assert_eq(result, expected)
        assert str(result.dtype) == dtype

    def test_writeable(self):
        """Test WRITEABLE requirement."""
        n = np.array([1, 2, 3], dtype="float64")
        r = rp.asarray(n)
        result = rp.require(r, requirements="W")
        expected = np.require(n, requirements="W")
        assert_eq(result, expected)


class TestCopyto:
    """Tests for copyto function."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        """Basic copy operation."""
        src_n = np.array([1, 2, 3], dtype=dtype)
        src_r = rp.asarray(src_n)
        dst_n = np.zeros(3, dtype=dtype)
        dst_r = rp.zeros(3, dtype=dtype)
        np.copyto(dst_n, src_n)
        rp.copyto(dst_r, src_r)
        assert_eq(dst_r, dst_n)

    def test_broadcast(self):
        """Copyto with broadcasting."""
        src_n = np.array([1, 2, 3], dtype="float64")
        src_r = rp.asarray(src_n)
        dst_n = np.zeros((2, 3), dtype="float64")
        dst_r = rp.zeros((2, 3), dtype="float64")
        np.copyto(dst_n, src_n)
        rp.copyto(dst_r, src_r)
        assert_eq(dst_r, dst_n)

    def test_with_where(self):
        """Copyto with where mask."""
        src_n = np.array([10, 20, 30, 40, 50], dtype="float64")
        src_r = rp.asarray(src_n)
        dst_n = np.zeros(5, dtype="float64")
        dst_r = rp.zeros(5, dtype="float64")
        mask_n = np.array([True, False, True, False, True])
        mask_r = rp.asarray(mask_n)
        np.copyto(dst_n, src_n, where=mask_n)
        rp.copyto(dst_r, src_r, where=mask_r)
        assert_eq(dst_r, dst_n)

    def test_casting(self):
        """Test casting parameter."""
        src_n = np.array([1, 2, 3], dtype="int32")
        src_r = rp.asarray(src_n)
        dst_n = np.zeros(3, dtype="float64")
        dst_r = rp.zeros(3, dtype="float64")
        np.copyto(dst_n, src_n, casting="safe")
        rp.copyto(dst_r, src_r, casting="safe")
        assert_eq(dst_r, dst_n)


class TestBroadcastShapes:
    """Tests for broadcast_shapes function."""

    def test_basic(self):
        """Basic broadcast shapes."""
        result = rp.broadcast_shapes((3, 1), (1, 4))
        expected = np.broadcast_shapes((3, 1), (1, 4))
        assert result == expected

    def test_multiple(self):
        """Multiple shapes."""
        result = rp.broadcast_shapes((2, 3, 4), (3, 1))
        expected = np.broadcast_shapes((2, 3, 4), (3, 1))
        assert result == expected

    def test_three_shapes(self):
        """Three shapes."""
        result = rp.broadcast_shapes((5, 1), (1, 6), (5, 6))
        expected = np.broadcast_shapes((5, 1), (1, 6), (5, 6))
        assert result == expected

    def test_single_shape(self):
        """Single shape returns itself."""
        result = rp.broadcast_shapes((3, 4))
        expected = np.broadcast_shapes((3, 4))
        assert result == expected

    def test_incompatible_raises(self):
        """Incompatible shapes should raise."""
        with pytest.raises(ValueError):
            rp.broadcast_shapes((3, 4), (5, 6))


class TestConcat:
    """Tests for concat (NumPy 2.0 alias for concatenate)."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_1d(self, dtype):
        """Concatenate 1D arrays."""
        a_n = np.array([1, 2], dtype=dtype)
        b_n = np.array([3, 4], dtype=dtype)
        a_r = rp.asarray(a_n)
        b_r = rp.asarray(b_n)
        result = rp.concat([a_r, b_r])
        expected = np.concat([a_n, b_n])
        assert_eq(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_with_axis(self, dtype):
        """Concatenate with axis."""
        a_n = np.array([[1, 2], [3, 4]], dtype=dtype)
        b_n = np.array([[5, 6]], dtype=dtype)
        a_r = rp.asarray(a_n)
        b_r = rp.asarray(b_n)
        result = rp.concat([a_r, b_r], axis=0)
        expected = np.concat([a_n, b_n], axis=0)
        assert_eq(result, expected)


class TestPermuteDims:
    """Tests for permute_dims (NumPy 2.0 alias for transpose)."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_2d(self, dtype):
        """Transpose 2D array."""
        n = np.array([[1, 2], [3, 4], [5, 6]], dtype=dtype)
        r = rp.asarray(n)
        result = rp.permute_dims(r, (1, 0))
        expected = np.permute_dims(n, (1, 0))
        assert_eq(result, expected)
        assert result.shape == expected.shape

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_3d(self, dtype):
        """Permute 3D array."""
        n = np.arange(24, dtype=dtype).reshape(2, 3, 4)
        r = rp.asarray(n)
        result = rp.permute_dims(r, (2, 0, 1))
        expected = np.permute_dims(n, (2, 0, 1))
        assert_eq(result, expected)
        assert result.shape == expected.shape


class TestMatrixTranspose:
    """Tests for matrix_transpose (swaps last two axes)."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_2d(self, dtype):
        """Transpose 2D array."""
        n = np.array([[1, 2], [3, 4]], dtype=dtype)
        r = rp.asarray(n)
        result = rp.matrix_transpose(r)
        expected = np.matrix_transpose(n)
        assert_eq(result, expected)
        assert result.shape == expected.shape

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_3d(self, dtype):
        """matrix_transpose on 3D swaps last two axes."""
        n = np.arange(24, dtype=dtype).reshape(2, 3, 4)
        r = rp.asarray(n)
        result = rp.matrix_transpose(r)
        expected = np.matrix_transpose(n)
        assert_eq(result, expected)
        assert result.shape == expected.shape

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_4d(self, dtype):
        """matrix_transpose on 4D swaps last two axes."""
        n = np.arange(120, dtype=dtype).reshape(2, 3, 4, 5)
        r = rp.asarray(n)
        result = rp.matrix_transpose(r)
        expected = np.matrix_transpose(n)
        assert_eq(result, expected)
        assert result.shape == expected.shape

    def test_1d_raises(self):
        """1D array should raise ValueError."""
        n = np.array([1, 2, 3])
        r = rp.asarray(n)
        with pytest.raises(ValueError):
            rp.matrix_transpose(r)
