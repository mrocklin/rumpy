"""Tests for sorting operations.

Parametrizes over operations for concise comprehensive coverage.
See designs/testing.md for testing philosophy.
"""

import numpy as np
import pytest

import rumpy as rp
from conftest import CORE_SHAPES, NUMERIC_DTYPES
from helpers import assert_eq, make_pair

# === Sorting operation categories ===

# Basic sorting operations (return sorted array or indices)
SORT_OPS = ["sort", "argsort"]

# Partial sorting operations (kth element)
PARTITION_OPS = ["partition", "argpartition"]


# === Parametrized tests by category ===


class TestSortOps:
    """Test basic sort and argsort operations."""

    @pytest.mark.parametrize("op", SORT_OPS)
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_dtypes(self, op, dtype):
        """Test all numeric dtypes."""
        n = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(getattr(rp, op)(r), getattr(np, op)(n))

    @pytest.mark.parametrize("op", SORT_OPS)
    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, op, shape):
        """Test different array shapes with default axis."""
        r, n = make_pair(shape, "float64")
        assert_eq(getattr(rp, op)(r), getattr(np, op)(n))

    @pytest.mark.parametrize("op", SORT_OPS)
    def test_axis_last(self, op):
        """Test sorting along last axis (axis=-1)."""
        r, n = make_pair((3, 4), "float64")
        assert_eq(getattr(rp, op)(r, axis=-1), getattr(np, op)(n, axis=-1))

    @pytest.mark.parametrize("op", SORT_OPS)
    def test_axis_0(self, op):
        """Test sorting along first axis."""
        r, n = make_pair((3, 4), "float64")
        assert_eq(getattr(rp, op)(r, axis=0), getattr(np, op)(n, axis=0))

    @pytest.mark.parametrize("op", SORT_OPS)
    def test_axis_1(self, op):
        """Test sorting along axis 1."""
        r, n = make_pair((3, 4), "float64")
        assert_eq(getattr(rp, op)(r, axis=1), getattr(np, op)(n, axis=1))

    @pytest.mark.parametrize("op", SORT_OPS)
    def test_axis_none(self, op):
        """Test sorting with axis=None (flattened)."""
        r, n = make_pair((3, 4), "float64")
        assert_eq(getattr(rp, op)(r, axis=None), getattr(np, op)(n, axis=None))

    @pytest.mark.parametrize("op", SORT_OPS)
    def test_3d_axis_0(self, op):
        """Test 3D sorting along axis 0 (positive axis only for 3D)."""
        r, n = make_pair((2, 3, 4), "float64")
        assert_eq(getattr(rp, op)(r, axis=0), getattr(np, op)(n, axis=0))

    @pytest.mark.parametrize("op", SORT_OPS)
    def test_3d_axis_1(self, op):
        """Test 3D sorting along axis 1 (positive axis only for 3D)."""
        r, n = make_pair((2, 3, 4), "float64")
        assert_eq(getattr(rp, op)(r, axis=1), getattr(np, op)(n, axis=1))

    @pytest.mark.parametrize("op", SORT_OPS)
    def test_3d_axis_2(self, op):
        """Test 3D sorting along axis 2 (positive axis only for 3D)."""
        r, n = make_pair((2, 3, 4), "float64")
        assert_eq(getattr(rp, op)(r, axis=2), getattr(np, op)(n, axis=2))

    @pytest.mark.parametrize("op", SORT_OPS)
    def test_already_sorted(self, op):
        """Test sorting already sorted array."""
        n = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(getattr(rp, op)(r), getattr(np, op)(n))

    @pytest.mark.parametrize("op", SORT_OPS)
    def test_reverse_sorted(self, op):
        """Test sorting reverse sorted array."""
        n = np.array([5, 4, 3, 2, 1], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(getattr(rp, op)(r), getattr(np, op)(n))

    @pytest.mark.parametrize("op", SORT_OPS)
    def test_duplicates(self, op):
        """Test sorting array with duplicates."""
        n = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(getattr(rp, op)(r), getattr(np, op)(n))

    @pytest.mark.parametrize("op", SORT_OPS)
    def test_single_element(self, op):
        """Test sorting single element array."""
        n = np.array([42], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(getattr(rp, op)(r), getattr(np, op)(n))

    @pytest.mark.parametrize("op", SORT_OPS)
    def test_empty(self, op):
        """Test sorting empty array."""
        n = np.array([], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(getattr(rp, op)(r), getattr(np, op)(n))

    def test_sort_with_nans(self):
        """Test sort with NaN values - verify it doesn't crash."""
        n = np.array([3.0, np.nan, 1.0, 2.0, np.nan])
        r = rp.asarray(n)
        # NaN handling may differ from numpy, just verify it doesn't crash
        result = rp.sort(r)
        assert result.size == 5

    def test_argsort_with_nans(self):
        """Test argsort with NaN values."""
        n = np.array([3.0, np.nan, 1.0, 2.0, np.nan])
        r = rp.asarray(n)
        # NaN ordering may differ, just verify it doesn't crash
        result = rp.argsort(r)
        assert result.size == 5


class TestPartitionOps:
    """Test partition and argpartition operations."""

    @pytest.mark.parametrize("op", PARTITION_OPS)
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_dtypes(self, op, dtype):
        """Test all numeric dtypes."""
        n = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=dtype)
        r = rp.asarray(n)
        # Test with kth in middle
        kth = 3
        r_result = getattr(rp, op)(r, kth)
        n_result = getattr(np, op)(n, kth)
        # For partition/argpartition, kth element position is deterministic
        if op == "partition":
            assert r_result[kth] == n_result[kth]
        else:  # argpartition
            assert r[r_result[kth]] == n[n_result[kth]]

    @pytest.mark.parametrize("op", PARTITION_OPS)
    def test_kth_positions(self, op):
        """Test different kth positions."""
        n = np.array([5, 2, 8, 1, 9, 3], dtype=np.float64)
        r = rp.asarray(n)
        for kth in range(len(n)):
            r_result = getattr(rp, op)(r, kth)
            n_result = getattr(np, op)(n, kth)
            if op == "partition":
                assert r_result[kth] == n_result[kth], f"Mismatch at kth={kth}"
            else:  # argpartition
                assert r[r_result[kth]] == n[n_result[kth]], f"Mismatch at kth={kth}"

    @pytest.mark.parametrize("op", PARTITION_OPS)
    def test_axis_last(self, op):
        """Test partition along last axis."""
        r, n = make_pair((3, 4), "float64")
        assert_eq(getattr(rp, op)(r, 1, axis=-1), getattr(np, op)(n, 1, axis=-1))

    @pytest.mark.parametrize("op", PARTITION_OPS)
    def test_axis_0(self, op):
        """Test partition along axis 0."""
        r, n = make_pair((3, 4), "float64")
        assert_eq(getattr(rp, op)(r, 1, axis=0), getattr(np, op)(n, 1, axis=0))

    @pytest.mark.parametrize("op", PARTITION_OPS)
    def test_axis_1(self, op):
        """Test partition along axis 1."""
        r, n = make_pair((3, 4), "float64")
        assert_eq(getattr(rp, op)(r, 1, axis=1), getattr(np, op)(n, 1, axis=1))

    @pytest.mark.parametrize("op", PARTITION_OPS)
    def test_axis_none(self, op):
        """Test partition with axis=None (flattened)."""
        r, n = make_pair((3, 4), "float64")
        result_r = getattr(rp, op)(r, 5, axis=None)
        result_n = getattr(np, op)(n, 5, axis=None)
        if op == "partition":
            assert result_r[5] == result_n[5]
        else:  # argpartition
            assert r.ravel()[result_r[5]] == n.ravel()[result_n[5]]

    @pytest.mark.parametrize("op", PARTITION_OPS)
    def test_3d_axis_0(self, op):
        """Test 3D partition along axis 0 (positive axis only for 3D)."""
        r, n = make_pair((2, 3, 4), "float64")
        assert_eq(getattr(rp, op)(r, 0, axis=0), getattr(np, op)(n, 0, axis=0))

    @pytest.mark.parametrize("op", PARTITION_OPS)
    def test_3d_axis_1(self, op):
        """Test 3D partition along axis 1 (positive axis only for 3D)."""
        r, n = make_pair((2, 3, 4), "float64")
        assert_eq(getattr(rp, op)(r, 1, axis=1), getattr(np, op)(n, 1, axis=1))

    @pytest.mark.parametrize("op", PARTITION_OPS)
    def test_3d_axis_2(self, op):
        """Test 3D partition along axis 2 (positive axis only for 3D)."""
        r, n = make_pair((2, 3, 4), "float64")
        assert_eq(getattr(rp, op)(r, 1, axis=2), getattr(np, op)(n, 1, axis=2))

    @pytest.mark.parametrize("op", PARTITION_OPS)
    def test_single_element(self, op):
        """Test partition of single element."""
        n = np.array([42], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(getattr(rp, op)(r, 0), getattr(np, op)(n, 0))

    def test_partition_with_nans(self):
        """Test partition with NaN values."""
        n = np.array([3.0, np.nan, 1.0, 2.0])
        r = rp.asarray(n)
        # Just verify it doesn't crash
        result = rp.partition(r, 1)
        assert result.size == 4


class TestUnique:
    """Test unique operation."""

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_dtypes(self, dtype):
        """Test all numeric dtypes."""
        n = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(rp.unique(r), np.unique(n))

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test different array shapes (unique always flattens)."""
        r, n = make_pair(shape, "float64")
        assert_eq(rp.unique(r), np.unique(n))

    def test_already_unique(self):
        """Test unique on array with no duplicates."""
        n = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.unique(r), np.unique(n))

    def test_all_same(self):
        """Test unique on array with all same values."""
        n = np.array([7, 7, 7, 7, 7], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.unique(r), np.unique(n))

    def test_single_element(self):
        """Test unique on single element array."""
        n = np.array([42], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.unique(r), np.unique(n))

    def test_empty(self):
        """Test unique on empty array."""
        n = np.array([], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.unique(r), np.unique(n))

    def test_with_nans(self):
        """Test unique with NaN values - verify it doesn't crash."""
        n = np.array([1.0, np.nan, 2.0, np.nan, 1.0])
        r = rp.asarray(n)
        # NaN handling may differ from numpy, just verify it doesn't crash
        result = rp.unique(r)
        assert result.size > 0

    def test_negative_values(self):
        """Test unique with negative values."""
        n = np.array([-3, -1, -3, 0, 1, -1, 2], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.unique(r), np.unique(n))

    def test_multidimensional(self):
        """Test unique on multidimensional array (should flatten)."""
        n = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.unique(r), np.unique(n))


class TestLexsort:
    """Test lexsort operation."""

    def test_basic_two_keys(self):
        """Test basic lexsort with two keys."""
        # Last key is primary sort key
        k1 = np.array([2, 1, 2, 1])
        k2 = np.array([1, 2, 2, 1])
        r = rp.lexsort((rp.asarray(k1), rp.asarray(k2)))
        n = np.lexsort((k1, k2))
        assert_eq(r, n)

    def test_single_key(self):
        """Test lexsort with single key (should behave like argsort)."""
        k = np.array([3, 1, 4, 1, 5])
        r = rp.lexsort((rp.asarray(k),))
        n = np.lexsort((k,))
        assert_eq(r, n)

    def test_three_keys(self):
        """Test lexsort with three keys."""
        k1 = np.array([1, 2, 1, 2])  # tertiary
        k2 = np.array([1, 1, 2, 2])  # secondary
        k3 = np.array([1, 1, 1, 1])  # primary (all same)
        r = rp.lexsort((rp.asarray(k1), rp.asarray(k2), rp.asarray(k3)))
        n = np.lexsort((k1, k2, k3))
        assert_eq(r, n)

    def test_key_priority(self):
        """Test that last key is primary sort key."""
        primary = np.array([2, 1, 2, 1])
        secondary = np.array([1, 1, 2, 2])
        r = rp.lexsort((rp.asarray(secondary), rp.asarray(primary)))
        n = np.lexsort((secondary, primary))
        assert_eq(r, n)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_dtypes(self, dtype):
        """Test different numeric dtypes."""
        k1 = np.array([2, 1, 2, 1], dtype=dtype)
        k2 = np.array([1, 2, 2, 1], dtype=dtype)
        r = rp.lexsort((rp.asarray(k1), rp.asarray(k2)))
        n = np.lexsort((k1, k2))
        assert_eq(r, n)

    def test_float_keys(self):
        """Test lexsort with float keys."""
        k1 = np.array([1.5, 2.5, 1.5, 2.5])
        k2 = np.array([0.1, 0.1, 0.2, 0.2])
        r = rp.lexsort((rp.asarray(k1), rp.asarray(k2)))
        n = np.lexsort((k1, k2))
        assert_eq(r, n)

    def test_all_duplicates(self):
        """Test lexsort with all duplicate values."""
        k1 = np.array([1, 1, 1, 1])
        k2 = np.array([2, 2, 2, 2])
        r = rp.lexsort((rp.asarray(k1), rp.asarray(k2)))
        n = np.lexsort((k1, k2))
        # All same, should maintain stable order
        assert_eq(r, n)

    def test_empty_keys(self):
        """Test lexsort with empty keys."""
        k = np.array([])
        r = rp.lexsort((rp.asarray(k),))
        n = np.lexsort((k,))
        assert r.size == 0
        assert n.size == 0

    def test_single_element_keys(self):
        """Test lexsort with single element."""
        k1 = np.array([5])
        r = rp.lexsort((rp.asarray(k1),))
        n = np.lexsort((k1,))
        assert_eq(r, n)

    def test_negative_values(self):
        """Test lexsort with negative values in keys."""
        k1 = np.array([-2, -1, -2, -1])
        k2 = np.array([-1, -2, -2, -1])
        r = rp.lexsort((rp.asarray(k1), rp.asarray(k2)))
        n = np.lexsort((k1, k2))
        assert_eq(r, n)

    def test_mixed_positive_negative(self):
        """Test lexsort with mixed positive and negative values."""
        k1 = np.array([2, -1, 2, -1])
        k2 = np.array([1, 2, -2, 1])
        r = rp.lexsort((rp.asarray(k1), rp.asarray(k2)))
        n = np.lexsort((k1, k2))
        assert_eq(r, n)


class TestSortingStability:
    """Test stable sorting behavior."""

    def test_sort_stable(self):
        """Test that sort is stable (equal elements maintain relative order)."""
        # Create array with duplicates
        n = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3], dtype=np.float64)
        r = rp.asarray(n)
        # For basic sort, we can't easily test stability without tracking indices
        # So we just verify the result matches numpy (which is stable)
        assert_eq(rp.sort(r), np.sort(n))

    def test_argsort_stable(self):
        """Test that argsort maintains stable order for equal elements."""
        # Array with duplicates at specific positions
        n = np.array([1, 2, 1, 2, 1], dtype=np.float64)
        r = rp.asarray(n)
        r_idx = rp.argsort(r)
        n_idx = np.argsort(n)
        assert_eq(r_idx, n_idx)


class TestUniqueExtensions:
    """Test NumPy 2.0+ unique extension functions."""

    def test_unique_values_basic(self):
        """Test basic unique_values.

        Note: We compare against np.unique since numpy's unique_values in 2.3.5
        has unexpected ordering behavior, while unique_values documentation says
        it should return sorted values like np.unique.
        """
        n = np.array([1, 1, 2])
        r = rp.asarray(n)
        assert_eq(rp.unique_values(r), np.unique(n))

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_unique_values_dtypes(self, dtype):
        """Test unique_values with all numeric dtypes."""
        n = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3], dtype=dtype)
        r = rp.asarray(n)
        # Compare against np.unique which returns sorted values
        assert_eq(rp.unique_values(r), np.unique(n))

    def test_unique_values_empty(self):
        """Test unique_values on empty array."""
        n = np.array([], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.unique_values(r), np.unique(n))

    def test_unique_counts_basic(self):
        """Test basic unique_counts."""
        n = np.array([1, 1, 2])
        r = rp.asarray(n)
        r_result = rp.unique_counts(r)
        n_result = np.unique_counts(n)
        assert_eq(r_result.values, n_result.values)
        assert_eq(r_result.counts, n_result.counts)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_unique_counts_dtypes(self, dtype):
        """Test unique_counts with all numeric dtypes."""
        n = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3], dtype=dtype)
        r = rp.asarray(n)
        r_result = rp.unique_counts(r)
        n_result = np.unique_counts(n)
        assert_eq(r_result.values, n_result.values)
        assert_eq(r_result.counts, n_result.counts)

    def test_unique_counts_empty(self):
        """Test unique_counts on empty array."""
        n = np.array([], dtype=np.float64)
        r = rp.asarray(n)
        r_result = rp.unique_counts(r)
        n_result = np.unique_counts(n)
        assert r_result.values.size == 0
        assert r_result.counts.size == 0

    def test_unique_counts_all_same(self):
        """Test unique_counts on array with all same values."""
        n = np.array([7, 7, 7, 7, 7], dtype=np.float64)
        r = rp.asarray(n)
        r_result = rp.unique_counts(r)
        n_result = np.unique_counts(n)
        assert_eq(r_result.values, n_result.values)
        assert_eq(r_result.counts, n_result.counts)

    def test_unique_inverse_basic(self):
        """Test basic unique_inverse."""
        n = np.array([1, 1, 2])
        r = rp.asarray(n)
        r_result = rp.unique_inverse(r)
        n_result = np.unique_inverse(n)
        assert_eq(r_result.values, n_result.values)
        assert_eq(r_result.inverse_indices, n_result.inverse_indices)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_unique_inverse_dtypes(self, dtype):
        """Test unique_inverse with all numeric dtypes."""
        n = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3], dtype=dtype)
        r = rp.asarray(n)
        r_result = rp.unique_inverse(r)
        n_result = np.unique_inverse(n)
        assert_eq(r_result.values, n_result.values)
        assert_eq(r_result.inverse_indices, n_result.inverse_indices)

    def test_unique_inverse_empty(self):
        """Test unique_inverse on empty array."""
        n = np.array([], dtype=np.float64)
        r = rp.asarray(n)
        r_result = rp.unique_inverse(r)
        n_result = np.unique_inverse(n)
        assert r_result.values.size == 0
        assert r_result.inverse_indices.size == 0

    def test_unique_inverse_reconstruction(self):
        """Test that inverse indices can reconstruct the original."""
        n = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3], dtype=np.float64)
        r = rp.asarray(n)
        r_result = rp.unique_inverse(r)
        # Reconstruct original array
        reconstructed = np.array(r_result.values)[np.array(r_result.inverse_indices)]
        assert_eq(rp.asarray(reconstructed), r)

    def test_unique_all_basic(self):
        """Test basic unique_all."""
        n = np.array([1, 1, 2])
        r = rp.asarray(n)
        r_result = rp.unique_all(r)
        n_result = np.unique_all(n)
        assert_eq(r_result.values, n_result.values)
        assert_eq(r_result.indices, n_result.indices)
        assert_eq(r_result.inverse_indices, n_result.inverse_indices)
        assert_eq(r_result.counts, n_result.counts)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_unique_all_dtypes(self, dtype):
        """Test unique_all with all numeric dtypes."""
        n = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3], dtype=dtype)
        r = rp.asarray(n)
        r_result = rp.unique_all(r)
        n_result = np.unique_all(n)
        assert_eq(r_result.values, n_result.values)
        assert_eq(r_result.indices, n_result.indices)
        assert_eq(r_result.inverse_indices, n_result.inverse_indices)
        assert_eq(r_result.counts, n_result.counts)

    def test_unique_all_empty(self):
        """Test unique_all on empty array."""
        n = np.array([], dtype=np.float64)
        r = rp.asarray(n)
        r_result = rp.unique_all(r)
        n_result = np.unique_all(n)
        assert r_result.values.size == 0
        assert r_result.indices.size == 0
        assert r_result.inverse_indices.size == 0
        assert r_result.counts.size == 0

    def test_unique_all_indices_first_occurrence(self):
        """Test that indices point to first occurrence."""
        n = np.array([3, 1, 4, 1, 5, 3], dtype=np.float64)
        r = rp.asarray(n)
        r_result = rp.unique_all(r)
        n_result = np.unique_all(n)
        assert_eq(r_result.indices, n_result.indices)

    def test_unique_all_multidimensional(self):
        """Test unique_all on multidimensional array (should flatten)."""
        n = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=np.float64)
        r = rp.asarray(n)
        r_result = rp.unique_all(r)
        n_result = np.unique_all(n)
        assert_eq(r_result.values, n_result.values)
        assert_eq(r_result.indices, n_result.indices)
        assert_eq(r_result.inverse_indices, n_result.inverse_indices)
        assert_eq(r_result.counts, n_result.counts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
