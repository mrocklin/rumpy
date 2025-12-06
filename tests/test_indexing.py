"""Comprehensive tests for array indexing against numpy.

Covers:
- Slices (start:stop:step, negative indices, negative steps)
- Integers (positive, negative)
- Python lists
- None/np.newaxis
- Boolean arrays
- Integer arrays (fancy indexing)
- Ellipsis (...)
- Combinations of the above
"""

import numpy as np
import pytest

import rumpy as rp
from helpers import assert_eq


class TestSlices:
    """Test slice indexing."""

    def test_basic_slice(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert_eq(r[2:7], n[2:7])

    def test_slice_from_start(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert_eq(r[:5], n[:5])

    def test_slice_to_end(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert_eq(r[5:], n[5:])

    def test_slice_all(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert_eq(r[:], n[:])

    def test_slice_with_positive_step(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert_eq(r[::2], n[::2])
        assert_eq(r[1::2], n[1::2])
        assert_eq(r[::3], n[::3])

    def test_slice_with_negative_step(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert_eq(r[::-1], n[::-1])
        assert_eq(r[::-2], n[::-2])
        assert_eq(r[8:2:-1], n[8:2:-1])

    def test_slice_negative_indices(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert_eq(r[-5:], n[-5:])
        assert_eq(r[:-3], n[:-3])
        assert_eq(r[-7:-2], n[-7:-2])

    def test_slice_out_of_bounds_clipped(self):
        """NumPy clips out-of-bounds slice indices."""
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert_eq(r[5:100], n[5:100])
        assert_eq(r[-100:5], n[-100:5])

    def test_slice_empty_result(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert_eq(r[5:2], n[5:2])  # Empty because start > stop with step=1


class TestSlices2D:
    """Test slice indexing on 2D arrays."""

    def test_slice_rows(self):
        r = rp.arange(20).reshape([4, 5])
        n = np.arange(20, dtype=np.float64).reshape(4, 5)
        assert_eq(r[1:3], n[1:3])

    def test_slice_both_axes(self):
        r = rp.arange(20).reshape([4, 5])
        n = np.arange(20, dtype=np.float64).reshape(4, 5)
        assert_eq(r[1:3, 2:4], n[1:3, 2:4])

    def test_slice_with_step_2d(self):
        r = rp.arange(20).reshape([4, 5])
        n = np.arange(20, dtype=np.float64).reshape(4, 5)
        assert_eq(r[::2, ::2], n[::2, ::2])

    def test_slice_negative_step_2d(self):
        r = rp.arange(20).reshape([4, 5])
        n = np.arange(20, dtype=np.float64).reshape(4, 5)
        assert_eq(r[::-1], n[::-1])
        assert_eq(r[:, ::-1], n[:, ::-1])
        assert_eq(r[::-1, ::-1], n[::-1, ::-1])


class TestSlices3D:
    """Test slice indexing on 3D arrays."""

    def test_slice_first_axis(self):
        r = rp.arange(24).reshape([2, 3, 4])
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        assert_eq(r[1:], n[1:])

    def test_slice_all_axes(self):
        r = rp.arange(24).reshape([2, 3, 4])
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        assert_eq(r[0:2, 1:3, 0:2], n[0:2, 1:3, 0:2])


class TestIntegerIndexing:
    """Test integer (scalar) indexing."""

    def test_single_int_1d(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        # Note: rumpy returns 0-d array, check values match
        assert float(np.asarray(r[0])) == float(n[0])
        assert float(np.asarray(r[5])) == float(n[5])
        assert float(np.asarray(r[-1])) == float(n[-1])

    def test_single_int_2d(self):
        r = rp.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r[0], n[0])
        assert_eq(r[1], n[1])
        assert_eq(r[-1], n[-1])

    def test_multi_int_2d(self):
        r = rp.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert float(np.asarray(r[1, 2])) == float(n[1, 2])
        assert float(np.asarray(r[-1, -1])) == float(n[-1, -1])

    def test_multi_int_3d(self):
        r = rp.arange(24).reshape([2, 3, 4])
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        assert float(np.asarray(r[0, 1, 2])) == float(n[0, 1, 2])
        assert float(np.asarray(r[-1, -1, -1])) == float(n[-1, -1, -1])

    def test_int_and_slice_mixed(self):
        r = rp.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r[1, :], n[1, :])
        assert_eq(r[:, 2], n[:, 2])
        assert_eq(r[1, 1:3], n[1, 1:3])


class TestListIndexing:
    """Test Python list indexing (should work like integer array)."""

    def test_list_1d(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert_eq(r[[0, 2, 4]], n[[0, 2, 4]])

    def test_list_with_duplicates(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert_eq(r[[0, 0, 1, 1]], n[[0, 0, 1, 1]])

    def test_list_reversed(self):
        r = rp.arange(5)
        n = np.arange(5, dtype=np.float64)
        assert_eq(r[[4, 3, 2, 1, 0]], n[[4, 3, 2, 1, 0]])

    def test_list_negative_indices(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert_eq(r[[-1, -2, -3]], n[[-1, -2, -3]])

    def test_list_2d_rows(self):
        r = rp.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r[[0, 2]], n[[0, 2]])


class TestNewaxis:
    """Test None/np.newaxis indexing."""

    def test_newaxis_front_1d(self):
        r = rp.arange(5)
        n = np.arange(5, dtype=np.float64)
        assert_eq(r[None], n[None])
        assert r[None].shape == n[None].shape

    def test_newaxis_back_1d(self):
        r = rp.arange(5)
        n = np.arange(5, dtype=np.float64)
        assert_eq(r[:, None], n[:, None])
        assert r[:, None].shape == n[:, None].shape

    def test_newaxis_multiple(self):
        r = rp.arange(5)
        n = np.arange(5, dtype=np.float64)
        assert_eq(r[None, :, None], n[None, :, None])
        assert r[None, :, None].shape == n[None, :, None].shape

    def test_newaxis_2d(self):
        r = rp.arange(6).reshape([2, 3])
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        assert_eq(r[None, :, :], n[None, :, :])
        assert_eq(r[:, None, :], n[:, None, :])
        assert_eq(r[:, :, None], n[:, :, None])

    def test_newaxis_with_slice(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert_eq(r[None, 2:7], n[None, 2:7])
        assert r[None, 2:7].shape == n[None, 2:7].shape


class TestBooleanIndexing:
    """Test boolean array indexing."""

    def test_bool_1d(self):
        r = rp.arange(5)
        n = np.arange(5, dtype=np.float64)
        mask_r = r > 2
        mask_n = n > 2
        assert_eq(r[mask_r], n[mask_n])

    def test_bool_all_true(self):
        r = rp.arange(5)
        n = np.arange(5, dtype=np.float64)
        mask_r = r >= 0
        mask_n = n >= 0
        assert_eq(r[mask_r], n[mask_n])

    def test_bool_all_false(self):
        r = rp.arange(5)
        n = np.arange(5, dtype=np.float64)
        mask_r = r < 0
        mask_n = n < 0
        assert_eq(r[mask_r], n[mask_n])

    def test_bool_2d(self):
        r = rp.arange(6).reshape([2, 3])
        n = np.arange(6, dtype=np.float64).reshape(2, 3)
        mask_r = r > 2
        mask_n = n > 2
        assert_eq(r[mask_r], n[mask_n])


class TestIntegerArrayIndexing:
    """Test integer array (fancy) indexing."""

    def test_array_1d(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        idx_r = rp.asarray([0, 2, 4])
        idx_n = np.array([0, 2, 4])
        assert_eq(r[idx_r], n[idx_n])

    def test_array_with_duplicates(self):
        r = rp.arange(5)
        n = np.arange(5, dtype=np.float64)
        idx_r = rp.asarray([0, 0, 1, 1, 2])
        idx_n = np.array([0, 0, 1, 1, 2])
        assert_eq(r[idx_r], n[idx_n])

    def test_array_reversed(self):
        r = rp.arange(5)
        n = np.arange(5, dtype=np.float64)
        idx_r = rp.asarray([4, 3, 2, 1, 0])
        idx_n = np.array([4, 3, 2, 1, 0])
        assert_eq(r[idx_r], n[idx_n])

    def test_array_negative_indices(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        idx_r = rp.asarray([-1, -2, -3])
        idx_n = np.array([-1, -2, -3])
        assert_eq(r[idx_r], n[idx_n])

    def test_array_2d_rows(self):
        r = rp.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        idx_r = rp.asarray([0, 2])
        idx_n = np.array([0, 2])
        assert_eq(r[idx_r], n[idx_n])


class TestEllipsis:
    """Test ellipsis (...) indexing."""

    def test_ellipsis_1d(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert_eq(r[...], n[...])

    def test_ellipsis_2d(self):
        r = rp.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r[...], n[...])

    def test_ellipsis_3d(self):
        r = rp.arange(24).reshape([2, 3, 4])
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        assert_eq(r[...], n[...])

    def test_ellipsis_with_int_front(self):
        r = rp.arange(24).reshape([2, 3, 4])
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        assert_eq(r[0, ...], n[0, ...])

    def test_ellipsis_with_int_back(self):
        r = rp.arange(24).reshape([2, 3, 4])
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        assert_eq(r[..., 0], n[..., 0])

    def test_ellipsis_with_slice_front(self):
        r = rp.arange(24).reshape([2, 3, 4])
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        assert_eq(r[0:1, ...], n[0:1, ...])

    def test_ellipsis_with_slice_back(self):
        r = rp.arange(24).reshape([2, 3, 4])
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        assert_eq(r[..., 1:3], n[..., 1:3])

    def test_ellipsis_in_middle(self):
        r = rp.arange(24).reshape([2, 3, 4])
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        assert_eq(r[0, ..., 1], n[0, ..., 1])


class TestCombinations:
    """Test combinations of indexing types."""

    def test_int_and_slice(self):
        r = rp.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r[1, 1:3], n[1, 1:3])
        assert_eq(r[0:2, 2], n[0:2, 2])

    def test_slice_and_newaxis(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert_eq(r[2:5, None], n[2:5, None])
        assert_eq(r[None, 2:5], n[None, 2:5])

    def test_int_and_newaxis(self):
        r = rp.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r[1, None, :], n[1, None, :])
        assert_eq(r[None, 1, :], n[None, 1, :])

    def test_ellipsis_and_newaxis(self):
        r = rp.arange(12).reshape([3, 4])
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(r[..., None], n[..., None])
        assert_eq(r[None, ...], n[None, ...])

    def test_ellipsis_and_slice(self):
        r = rp.arange(24).reshape([2, 3, 4])
        n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        assert_eq(r[..., 1:3], n[..., 1:3])
        assert_eq(r[0:1, ...], n[0:1, ...])


class TestEdgeCases:
    """Test edge cases and potential corner cases."""

    def test_empty_array_slice(self):
        r = rp.arange(0)
        n = np.arange(0, dtype=np.float64)
        assert_eq(r[:], n[:])

    def test_single_element_array(self):
        r = rp.arange(1)
        n = np.arange(1, dtype=np.float64)
        assert_eq(r[:], n[:])
        assert_eq(r[0:1], n[0:1])

    def test_negative_step_full_reverse(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert_eq(r[::-1], n[::-1])
        # Verify the reversal is correct
        assert float(np.asarray(r[::-1])[0]) == 9.0
        assert float(np.asarray(r[::-1])[-1]) == 0.0

    def test_large_step(self):
        r = rp.arange(10)
        n = np.arange(10, dtype=np.float64)
        assert_eq(r[::5], n[::5])
        assert_eq(r[::10], n[::10])
        assert_eq(r[::100], n[::100])


class TestDtypePreservation:
    """Test that indexing preserves dtype."""

    def test_slice_preserves_float32(self):
        r = rp.arange(10, dtype="float32")
        assert r[2:5].dtype == "float32"

    def test_slice_preserves_int32(self):
        r = rp.arange(10, dtype="int32")
        assert r[2:5].dtype == "int32"

    def test_fancy_index_preserves_dtype(self):
        r = rp.arange(10, dtype="float32")
        idx = rp.asarray([0, 2, 4])
        assert r[idx].dtype == "float32"

    def test_bool_index_preserves_dtype(self):
        r = rp.arange(10, dtype="float32")
        mask = r > 5
        assert r[mask].dtype == "float32"
