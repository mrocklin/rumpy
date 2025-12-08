"""Comprehensive tests for array indexing against numpy.

Covers:
- Integer indexing (single element, multiple indices)
- Slice indexing (start:stop:step, negative indices)
- Boolean indexing (mask arrays)
- Fancy indexing (integer arrays as indices)
- Negative indices, step values
- Out-of-bounds clipping behavior
- Mixed indexing (int + slice, newaxis, ellipsis)

See designs/testing.md for testing philosophy.
"""

import numpy as np
import pytest
import rumpy as rp
from conftest import CORE_SHAPES, FLOAT_DTYPES, NUMERIC_DTYPES
from helpers import assert_eq, make_pair


class TestIntegerIndexing:
    """Test integer (scalar) indexing."""

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_single_element_1d(self, dtype):
        r, n = make_pair((10,), dtype)
        # Note: rumpy returns 0-d array, numpy returns scalar
        assert float(np.asarray(r[5])) == float(n[5])
        assert float(np.asarray(r[0])) == float(n[0])
        assert float(np.asarray(r[-1])) == float(n[-1])

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_single_element_2d(self, dtype):
        r, n = make_pair((3, 4), dtype)
        assert_eq(r[0], n[0])  # Returns 1D array
        assert_eq(r[1], n[1])
        assert_eq(r[-1], n[-1])

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_multi_index_2d(self, dtype):
        r, n = make_pair((3, 4), dtype)
        assert float(np.asarray(r[1, 2])) == float(n[1, 2])
        assert float(np.asarray(r[-1, -1])) == float(n[-1, -1])
        assert float(np.asarray(r[0, 0])) == float(n[0, 0])

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_multi_index_3d(self, dtype):
        r, n = make_pair((2, 3, 4), dtype)
        assert float(np.asarray(r[0, 1, 2])) == float(n[0, 1, 2])
        assert float(np.asarray(r[-1, -1, -1])) == float(n[-1, -1, -1])


class TestSliceIndexing:
    """Test slice indexing with various patterns."""

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic_slice_1d(self, dtype):
        r, n = make_pair((10,), dtype)
        assert_eq(r[2:7], n[2:7])
        assert_eq(r[:5], n[:5])
        assert_eq(r[5:], n[5:])
        assert_eq(r[:], n[:])

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_slice_with_step(self, dtype):
        r, n = make_pair((10,), dtype)
        assert_eq(r[::2], n[::2])
        assert_eq(r[1::2], n[1::2])
        assert_eq(r[::3], n[::3])
        assert_eq(r[2:8:2], n[2:8:2])

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_slice_negative_step(self, dtype):
        r, n = make_pair((10,), dtype)
        assert_eq(r[::-1], n[::-1])
        assert_eq(r[::-2], n[::-2])
        assert_eq(r[8:2:-1], n[8:2:-1])
        assert_eq(r[::- 1], n[::-1])

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_slice_negative_indices(self, dtype):
        r, n = make_pair((10,), dtype)
        assert_eq(r[-5:], n[-5:])
        assert_eq(r[:-3], n[:-3])
        assert_eq(r[-7:-2], n[-7:-2])
        assert_eq(r[-1:-5:-1], n[-1:-5:-1])

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_slice_2d(self, dtype):
        r, n = make_pair((4, 5), dtype)
        assert_eq(r[1:3], n[1:3])
        assert_eq(r[1:3, 2:4], n[1:3, 2:4])
        assert_eq(r[::2, ::2], n[::2, ::2])
        assert_eq(r[::-1], n[::-1])
        assert_eq(r[:, ::-1], n[:, ::-1])

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_slice_3d(self, dtype):
        r, n = make_pair((2, 3, 4), dtype)
        assert_eq(r[1:], n[1:])
        assert_eq(r[0:2, 1:3, 0:2], n[0:2, 1:3, 0:2])
        assert_eq(r[::1, ::1, ::1], n[::1, ::1, ::1])

    def test_slice_out_of_bounds_clipped(self):
        """NumPy clips out-of-bounds slice indices."""
        r, n = make_pair((10,), "float64")
        assert_eq(r[5:100], n[5:100])
        assert_eq(r[-100:5], n[-100:5])
        assert_eq(r[-100:100], n[-100:100])

    def test_slice_empty_result(self):
        r, n = make_pair((10,), "float64")
        assert_eq(r[5:2], n[5:2])  # Empty because start > stop with step=1
        assert_eq(r[0:0], n[0:0])


class TestBooleanIndexing:
    """Test boolean array indexing."""

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_bool_mask_1d(self, dtype):
        r, n = make_pair((10,), dtype)
        mask_r = r > np.asarray(r)[len(np.asarray(r)) // 2]
        mask_n = n > n[len(n) // 2]
        assert_eq(r[mask_r], n[mask_n])

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_bool_all_true(self, dtype):
        r, n = make_pair((5,), dtype)
        mask_r = r >= 0
        mask_n = n >= 0
        assert_eq(r[mask_r], n[mask_n])

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_bool_all_false(self, dtype):
        r, n = make_pair((5,), dtype)
        mask_r = r < -100
        mask_n = n < -100
        assert_eq(r[mask_r], n[mask_n])

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_bool_2d(self, dtype):
        r, n = make_pair((3, 4), dtype)
        mask_r = r > 5
        mask_n = n > 5
        assert_eq(r[mask_r], n[mask_n])

    def test_bool_explicit_mask(self):
        """Test with explicit boolean mask."""
        r, n = make_pair((5,), "float64")
        mask_r = rp.asarray([True, False, True, False, True])
        mask_n = np.array([True, False, True, False, True])
        assert_eq(r[mask_r], n[mask_n])


class TestFancyIndexing:
    """Test integer array (fancy) indexing."""

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_integer_array_1d(self, dtype):
        r, n = make_pair((10,), dtype)
        idx_r = rp.asarray([0, 2, 4, 6, 8])
        idx_n = np.array([0, 2, 4, 6, 8])
        assert_eq(r[idx_r], n[idx_n])

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_list_indexing_1d(self, dtype):
        """Python lists should work like integer arrays."""
        r, n = make_pair((10,), dtype)
        assert_eq(r[[0, 2, 4]], n[[0, 2, 4]])
        assert_eq(r[[0, 0, 1, 1]], n[[0, 0, 1, 1]])

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_fancy_with_duplicates(self, dtype):
        r, n = make_pair((5,), dtype)
        idx_r = rp.asarray([0, 0, 1, 1, 2])
        idx_n = np.array([0, 0, 1, 1, 2])
        assert_eq(r[idx_r], n[idx_n])

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_fancy_reversed(self, dtype):
        r, n = make_pair((5,), dtype)
        idx_r = rp.asarray([4, 3, 2, 1, 0])
        idx_n = np.array([4, 3, 2, 1, 0])
        assert_eq(r[idx_r], n[idx_n])

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_fancy_negative_indices(self, dtype):
        r, n = make_pair((10,), dtype)
        idx_r = rp.asarray([-1, -2, -3])
        idx_n = np.array([-1, -2, -3])
        assert_eq(r[idx_r], n[idx_n])

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_fancy_2d_rows(self, dtype):
        r, n = make_pair((4, 5), dtype)
        idx_r = rp.asarray([0, 2, 3])
        idx_n = np.array([0, 2, 3])
        assert_eq(r[idx_r], n[idx_n])

    def test_fancy_empty_indices(self):
        r, n = make_pair((10,), "float64")
        idx_r = rp.asarray([], dtype="int64").reshape([0])
        idx_n = np.array([], dtype=np.int64).reshape([0])
        result_r = r[idx_r]
        result_n = n[idx_n]
        assert result_r.size == 0
        assert list(result_r.shape) == list(result_n.shape)


class TestNewaxis:
    """Test None/np.newaxis indexing."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_newaxis_1d(self, dtype):
        r, n = make_pair((5,), dtype)
        assert_eq(r[None], n[None])
        assert r[None].shape == n[None].shape
        assert_eq(r[:, None], n[:, None])
        assert r[:, None].shape == n[:, None].shape

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_newaxis_multiple(self, dtype):
        r, n = make_pair((5,), dtype)
        assert_eq(r[None, :, None], n[None, :, None])
        assert r[None, :, None].shape == n[None, :, None].shape

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_newaxis_2d(self, dtype):
        r, n = make_pair((2, 3), dtype)
        assert_eq(r[None, :, :], n[None, :, :])
        assert_eq(r[:, None, :], n[:, None, :])
        assert_eq(r[:, :, None], n[:, :, None])

    def test_newaxis_with_slice(self):
        r, n = make_pair((10,), "float64")
        assert_eq(r[None, 2:7], n[None, 2:7])
        assert r[None, 2:7].shape == n[None, 2:7].shape


class TestEllipsis:
    """Test ellipsis (...) indexing."""

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_ellipsis_basic(self, shape, dtype):
        r, n = make_pair(shape, dtype)
        assert_eq(r[...], n[...])

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_ellipsis_with_int(self, dtype):
        r, n = make_pair((2, 3, 4), dtype)
        assert_eq(r[0, ...], n[0, ...])
        assert_eq(r[..., 0], n[..., 0])

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_ellipsis_with_slice(self, dtype):
        r, n = make_pair((2, 3, 4), dtype)
        assert_eq(r[0:1, ...], n[0:1, ...])
        assert_eq(r[..., 1:3], n[..., 1:3])

    def test_ellipsis_in_middle(self):
        r, n = make_pair((2, 3, 4), "float64")
        assert_eq(r[0, ..., 1], n[0, ..., 1])


class TestMixedIndexing:
    """Test combinations of indexing types."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_int_and_slice(self, dtype):
        r, n = make_pair((3, 4), dtype)
        assert_eq(r[1, :], n[1, :])
        assert_eq(r[:, 2], n[:, 2])
        assert_eq(r[1, 1:3], n[1, 1:3])
        assert_eq(r[0:2, 2], n[0:2, 2])

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_slice_and_newaxis(self, dtype):
        r, n = make_pair((10,), dtype)
        assert_eq(r[2:5, None], n[2:5, None])
        assert_eq(r[None, 2:5], n[None, 2:5])

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_int_and_newaxis(self, dtype):
        r, n = make_pair((3, 4), dtype)
        assert_eq(r[1, None, :], n[1, None, :])
        assert_eq(r[None, 1, :], n[None, 1, :])

    def test_ellipsis_and_newaxis(self):
        r, n = make_pair((3, 4), "float64")
        assert_eq(r[..., None], n[..., None])
        assert_eq(r[None, ...], n[None, ...])


class TestEdgeCases:
    """Test edge cases and corner cases."""

    def test_empty_array_slice(self):
        r = rp.arange(0)
        n = np.arange(0, dtype=np.float64)
        assert_eq(r[:], n[:])

    def test_single_element_array(self):
        r = rp.arange(1)
        n = np.arange(1, dtype=np.float64)
        assert_eq(r[:], n[:])
        assert_eq(r[0:1], n[0:1])

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_large_step(self, dtype):
        r, n = make_pair((10,), dtype)
        assert_eq(r[::5], n[::5])
        assert_eq(r[::10], n[::10])
        assert_eq(r[::100], n[::100])

    def test_negative_step_full_reverse(self):
        r, n = make_pair((10,), "float64")
        assert_eq(r[::-1], n[::-1])
        # Verify the reversal is correct
        assert float(np.asarray(r[::-1])[0]) == float(n[::-1][0])

    @pytest.mark.parametrize("shape", [(0,), (0, 5), (5, 0)])
    def test_empty_shapes(self, shape):
        r = rp.zeros(shape)
        n = np.zeros(shape)
        assert_eq(r[:], n[:])
        assert_eq(r[...], n[...])


class TestDtypePreservation:
    """Test that indexing preserves dtype."""

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_slice_preserves_dtype(self, dtype):
        r, n = make_pair((10,), dtype)
        assert str(r[2:5].dtype) == str(n[2:5].dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_fancy_preserves_dtype(self, dtype):
        r, n = make_pair((10,), dtype)
        idx = rp.asarray([0, 2, 4])
        assert str(r[idx].dtype) == str(n[[0, 2, 4]].dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_bool_preserves_dtype(self, dtype):
        r, n = make_pair((10,), dtype)
        mask_r = r > np.asarray(r)[5]
        mask_n = n > n[5]
        assert str(r[mask_r].dtype) == str(n[mask_n].dtype)


class TestIndexingOperations:
    """Test indexing operation functions (take, compress, searchsorted)."""

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_take_1d(self, dtype):
        r, n = make_pair((10,), dtype)
        indices = [0, 2, 4, 6, 8]
        assert_eq(rp.take(r, indices), np.take(n, indices))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_take_2d_axis0(self, dtype):
        r, n = make_pair((3, 4), dtype)
        indices = [0, 2]
        assert_eq(rp.take(r, indices, axis=0), np.take(n, indices, axis=0))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_take_2d_axis1(self, dtype):
        r, n = make_pair((3, 4), dtype)
        indices = [0, 2, 3]
        assert_eq(rp.take(r, indices, axis=1), np.take(n, indices, axis=1))

    def test_take_negative_indices(self):
        r, n = make_pair((10,), "float64")
        indices = [-1, -2, -3]
        assert_eq(rp.take(r, indices), np.take(n, indices))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_compress_1d(self, dtype):
        r, n = make_pair((6,), dtype)
        condition = [True, False, True, False, True, False]
        assert_eq(rp.compress(condition, r), np.compress(condition, n))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_compress_2d(self, dtype):
        r, n = make_pair((3, 4), dtype)
        condition = [True, False, True]
        assert_eq(rp.compress(condition, r, axis=0), np.compress(condition, n, axis=0))

    def test_searchsorted_basic(self):
        r = rp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
        n = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        v_r = rp.asarray([0.5, 1.5, 2.5, 5.5])
        v_n = np.array([0.5, 1.5, 2.5, 5.5])
        assert_eq(rp.searchsorted(r, v_r), np.searchsorted(n, v_n))

    def test_searchsorted_scalar(self):
        r = rp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
        n = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert int(np.asarray(rp.searchsorted(r, 2.5))) == int(np.searchsorted(n, 2.5))


class TestSetitem:
    """Test slice assignment (__setitem__)."""

    def test_setitem_scalar_to_slice(self):
        """arr[2:5] = 0 should set elements to 0."""
        r = rp.ones(10)
        n = np.ones(10)
        r[2:5] = 0
        n[2:5] = 0
        assert_eq(r, n)

    def test_setitem_scalar_to_negative_slice(self):
        """arr[2:-2] = 0 should set middle elements."""
        r = rp.ones(10)
        n = np.ones(10)
        r[2:-2] = 0
        n[2:-2] = 0
        assert_eq(r, n)

    def test_setitem_scalar_to_stepped_slice(self):
        """arr[::2] = 0 should set every other element."""
        r = rp.ones(10)
        n = np.ones(10)
        r[::2] = 0
        n[::2] = 0
        assert_eq(r, n)

    def test_setitem_array_to_slice(self):
        """arr[2:5] = [7, 8, 9] should work."""
        r = rp.ones(10)
        n = np.ones(10)
        r[2:5] = rp.array([7.0, 8.0, 9.0])
        n[2:5] = np.array([7.0, 8.0, 9.0])
        assert_eq(r, n)

    def test_setitem_2d_row(self):
        """arr[1, :] = 0 should set an entire row."""
        r = rp.ones((3, 4))
        n = np.ones((3, 4))
        r[1, :] = 0
        n[1, :] = 0
        assert_eq(r, n)

    def test_setitem_2d_col(self):
        """arr[:, 2] = 0 should set an entire column."""
        r = rp.ones((3, 4))
        n = np.ones((3, 4))
        r[:, 2] = 0
        n[:, 2] = 0
        assert_eq(r, n)

    def test_setitem_2d_subarray(self):
        """arr[1:3, 1:3] = 0 should set a 2x2 block."""
        r = rp.ones((4, 4))
        n = np.ones((4, 4))
        r[1:3, 1:3] = 0
        n[1:3, 1:3] = 0
        assert_eq(r, n)

    def test_setitem_single_element(self):
        """arr[3] = 99 should set single element."""
        r = rp.ones(10)
        n = np.ones(10)
        r[3] = 99.0
        n[3] = 99.0
        assert_eq(r, n)

    def test_setitem_single_element_2d(self):
        """arr[1, 2] = 99 should set single element in 2D array."""
        r = rp.ones((3, 4))
        n = np.ones((3, 4))
        r[1, 2] = 99.0
        n[1, 2] = 99.0
        assert_eq(r, n)
