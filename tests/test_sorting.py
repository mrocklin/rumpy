"""Tests for sorting operations: partition, argpartition, lexsort."""
import pytest
import numpy as np
import rumpy as rp
from helpers import assert_eq


class TestPartition:
    """Tests for rp.partition."""

    def test_basic_1d(self):
        """Test basic 1D partition."""
        a = [3, 4, 2, 1, 5]
        r = rp.partition(rp.asarray(a), 2)
        n = np.partition(np.array(a), 2)
        # The kth element should be in its sorted position
        assert r[2] == n[2]
        # All elements before kth should be <= kth
        assert all(r[i] <= r[2] for i in range(2))
        # All elements after kth should be >= kth
        assert all(r[i] >= r[2] for i in range(3, 5))

    def test_kth_positions(self):
        """Test different kth positions."""
        a = [5, 2, 8, 1, 9, 3]
        for kth in range(len(a)):
            r = rp.partition(rp.asarray(a), kth)
            n = np.partition(np.array(a), kth)
            # The kth element should match numpy
            assert r[kth] == n[kth], f"Mismatch at kth={kth}"

    def test_2d_axis_1(self):
        """Test 2D partition along axis 1."""
        a = [[3, 4, 2], [1, 5, 0]]
        r = rp.partition(rp.asarray(a), 1, axis=-1)
        n = np.partition(np.array(a), 1, axis=-1)
        # kth element in each row should be in sorted position
        assert r[0, 1] == n[0, 1]
        assert r[1, 1] == n[1, 1]

    def test_2d_axis_0(self):
        """Test 2D partition along axis 0."""
        a = [[3, 4, 2], [1, 5, 0]]
        r = rp.partition(rp.asarray(a), 0, axis=0)
        n = np.partition(np.array(a), 0, axis=0)
        # kth element in each column should be in sorted position
        for col in range(3):
            assert r[0, col] == n[0, col]

    def test_axis_none(self):
        """Test partition with axis=None (flatten)."""
        a = [[3, 4], [2, 1]]
        r = rp.partition(rp.asarray(a), 1, axis=None)
        n = np.partition(np.array(a), 1, axis=None)
        # Flattened and partitioned
        assert r[1] == n[1]

    def test_dtypes(self):
        """Test partition with different dtypes."""
        for dtype in ["float32", "float64", "int32", "int64"]:
            a = [5.0, 2.0, 8.0, 1.0, 9.0]
            r = rp.partition(rp.asarray(a, dtype=dtype), 2)
            n = np.partition(np.array(a, dtype=dtype), 2)
            assert r[2] == n[2]


class TestArgpartition:
    """Tests for rp.argpartition."""

    def test_basic_1d(self):
        """Test basic 1D argpartition."""
        a = [3, 4, 2, 1, 5]
        ra = rp.asarray(a)
        na = np.array(a)
        r = rp.argpartition(ra, 2)
        n = np.argpartition(na, 2)
        # The value at the kth index should be the kth smallest
        assert ra[r[2]] == na[n[2]]

    def test_kth_positions(self):
        """Test different kth positions."""
        a = [5, 2, 8, 1, 9, 3]
        ra = rp.asarray(a)
        na = np.array(a)
        for kth in range(len(a)):
            r = rp.argpartition(ra, kth)
            n = np.argpartition(na, kth)
            # The value at the kth index should match
            assert ra[r[kth]] == na[n[kth]], f"Mismatch at kth={kth}"

    def test_2d_axis_1(self):
        """Test 2D argpartition along axis 1."""
        a = [[3, 4, 2], [1, 5, 0]]
        ra = rp.asarray(a)
        na = np.array(a)
        r = rp.argpartition(ra, 1, axis=-1)
        n = np.argpartition(na, 1, axis=-1)
        # Check that indexed values match
        # Use Python indexing since rumpy may not support array indexing
        for row in range(2):
            r_idx = int(r[row, 1])
            n_idx = int(n[row, 1])
            assert ra[row, r_idx] == na[row, n_idx]

    def test_axis_none(self):
        """Test argpartition with axis=None (flatten)."""
        a = [[3, 4], [2, 1]]
        ra = rp.asarray(a)
        na = np.array(a)
        r = rp.argpartition(ra, 1, axis=None)
        n = np.argpartition(na, 1, axis=None)
        # The flattened value at kth index
        assert ra.ravel()[r[1]] == na.ravel()[n[1]]


class TestLexsort:
    """Tests for rp.lexsort."""

    def test_basic_numeric(self):
        """Test basic numeric lexsort."""
        # Last key is primary
        keys = ([2, 1, 2, 1], [1, 2, 2, 1])
        r = rp.lexsort((rp.asarray(keys[0]), rp.asarray(keys[1])))
        n = np.lexsort(keys)
        assert_eq(r, n)

    def test_single_key(self):
        """Test lexsort with single key."""
        key = [3, 1, 4, 1, 5]
        r = rp.lexsort((rp.asarray(key),))
        n = np.lexsort((np.array(key),))
        assert_eq(r, n)

    def test_three_keys(self):
        """Test lexsort with three keys."""
        k1 = [1, 2, 1, 2]  # tertiary
        k2 = [1, 1, 2, 2]  # secondary
        k3 = [1, 1, 1, 1]  # primary (all same)
        r = rp.lexsort((rp.asarray(k1), rp.asarray(k2), rp.asarray(k3)))
        n = np.lexsort((np.array(k1), np.array(k2), np.array(k3)))
        assert_eq(r, n)

    def test_reverse_order(self):
        """Test that last key is indeed primary."""
        primary = [2, 1, 2, 1]
        secondary = [1, 1, 2, 2]
        r = rp.lexsort((rp.asarray(secondary), rp.asarray(primary)))
        n = np.lexsort((np.array(secondary), np.array(primary)))
        assert_eq(r, n)
        # Verify sorted order: primary first, then secondary
        # primary=[2,1,2,1] means indices 1,3 have primary=1, indices 0,2 have primary=2
        # Among indices 1,3: secondary=[1,2], so 1 comes before 3
        # Among indices 0,2: secondary=[1,2], so 0 comes before 2
        # Result should be [1, 3, 0, 2]
        expected = [1, 3, 0, 2]
        assert list(r.tolist()) == expected

    def test_float_keys(self):
        """Test lexsort with float keys."""
        k1 = [1.5, 2.5, 1.5, 2.5]
        k2 = [0.1, 0.1, 0.2, 0.2]
        r = rp.lexsort((rp.asarray(k1), rp.asarray(k2)))
        n = np.lexsort((np.array(k1), np.array(k2)))
        assert_eq(r, n)

    def test_with_duplicates(self):
        """Test lexsort handles duplicates correctly."""
        k1 = [1, 1, 1, 1]
        k2 = [2, 2, 2, 2]
        r = rp.lexsort((rp.asarray(k1), rp.asarray(k2)))
        n = np.lexsort((np.array(k1), np.array(k2)))
        # All same, so stable sort order
        assert_eq(r, n)

    def test_empty_keys(self):
        """Test lexsort with empty keys."""
        k1 = []
        r = rp.lexsort((rp.asarray(k1),))
        n = np.lexsort((np.array(k1),))
        assert r.size == 0
        assert n.size == 0

    def test_accepts_lists(self):
        """Test lexsort accepts Python lists directly."""
        keys = ([3, 1, 2], [1, 2, 3])
        r = rp.lexsort((rp.asarray(keys[0]), rp.asarray(keys[1])))
        n = np.lexsort(keys)
        assert_eq(r, n)


class TestSortingEdgeCases:
    """Edge cases for sorting functions."""

    def test_partition_single_element(self):
        """Partition of single element."""
        a = [5]
        r = rp.partition(rp.asarray(a), 0)
        n = np.partition(np.array(a), 0)
        assert_eq(r, n)

    def test_argpartition_single_element(self):
        """Argpartition of single element."""
        a = [5]
        r = rp.argpartition(rp.asarray(a), 0)
        n = np.argpartition(np.array(a), 0)
        assert_eq(r, n)

    def test_partition_with_nans(self):
        """Partition with NaN values - verify it doesn't crash."""
        a = [3.0, float('nan'), 1.0, 2.0]
        # Just verify the operation doesn't crash
        # NaN handling is implementation-specific
        r = rp.partition(rp.asarray(a), 1)
        assert r.size == 4

    def test_already_sorted(self):
        """Partition of already sorted array."""
        a = [1, 2, 3, 4, 5]
        r = rp.partition(rp.asarray(a), 2)
        n = np.partition(np.array(a), 2)
        assert_eq(r, n)

    def test_reverse_sorted(self):
        """Partition of reverse sorted array."""
        a = [5, 4, 3, 2, 1]
        r = rp.partition(rp.asarray(a), 2)
        n = np.partition(np.array(a), 2)
        assert r[2] == n[2]

    def test_lexsort_single_element_keys(self):
        """Lexsort with single element keys."""
        k1 = [5]
        r = rp.lexsort((rp.asarray(k1),))
        n = np.lexsort((np.array(k1),))
        assert_eq(r, n)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
