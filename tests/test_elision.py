"""Tests for temporary array elision (in-place buffer reuse)."""

import numpy as np
import rumpy as rp
from helpers import assert_eq


class TestElisionCorrectness:
    """Verify chained operations produce correct results (elision is transparent)."""

    def test_chained_add(self):
        """Chain of adds produces correct result."""
        x = rp.arange(100000, dtype="float64")
        result = x + 1 + 2 + 3
        expected = np.arange(100000, dtype=np.float64) + 1 + 2 + 3
        assert_eq(result, expected)

    def test_chained_mul(self):
        """Chain of multiplies produces correct result."""
        x = rp.arange(100000, dtype="float64") + 1  # avoid zeros
        result = x * 2 * 3 * 4
        expected = (np.arange(100000, dtype=np.float64) + 1) * 2 * 3 * 4
        assert_eq(result, expected)

    def test_chained_sub(self):
        """Chain of subtracts produces correct result."""
        x = rp.arange(100000, dtype="float64")
        result = x - 1 - 2 - 3
        expected = np.arange(100000, dtype=np.float64) - 1 - 2 - 3
        assert_eq(result, expected)

    def test_chained_mixed_ops(self):
        """Mixed operations chain produces correct result."""
        x = rp.arange(100000, dtype="float64") + 1
        result = x * 2 + 3 - 4 * 2
        expected = (np.arange(100000, dtype=np.float64) + 1) * 2 + 3 - 4 * 2
        assert_eq(result, expected)

    def test_chained_array_ops(self):
        """Chain with array operands produces correct result."""
        x = rp.arange(100000, dtype="float64")
        y = rp.ones(100000, dtype="float64")
        result = x + y + y + y
        expected = np.arange(100000, dtype=np.float64) + 1 + 1 + 1
        assert_eq(result, expected)


class TestElisionReferencePreservation:
    """Verify that keeping references prevents modification of originals."""

    def test_reference_preserves_original(self):
        """Keeping a reference prevents elision - original unchanged."""
        x = rp.arange(100000, dtype="float64")
        y = x  # Keep extra reference
        _ = x + 1  # This should NOT modify x because y holds a reference
        expected = np.arange(100000, dtype=np.float64)
        assert_eq(x, expected)

    def test_multiple_references_preserve_original(self):
        """Multiple references prevent elision."""
        x = rp.arange(100000, dtype="float64")
        refs = [x, x, x]  # Multiple references
        _ = x + 1
        expected = np.arange(100000, dtype=np.float64)
        assert_eq(x, expected)
        del refs  # Keep refs alive until assertion

    def test_stored_in_list_preserves(self):
        """Array stored in list is preserved."""
        arrays = [rp.arange(100000, dtype="float64")]
        x = arrays[0]
        _ = x + 1
        expected = np.arange(100000, dtype=np.float64)
        assert_eq(arrays[0], expected)


class TestSmallArrays:
    """Verify small arrays work correctly (elision is size-gated)."""

    def test_small_array_chained(self):
        """Small arrays below threshold work correctly."""
        x = rp.array([1.0, 2.0, 3.0])
        result = x + 1 + 2 + 3
        expected = np.array([1.0, 2.0, 3.0]) + 1 + 2 + 3
        assert_eq(result, expected)

    def test_small_array_reference_safe(self):
        """Small arrays with references are safe."""
        x = rp.array([1.0, 2.0, 3.0])
        y = x
        _ = x + 1
        expected = np.array([1.0, 2.0, 3.0])
        assert_eq(x, expected)


class TestDtypeHandling:
    """Verify dtype-related edge cases."""

    def test_type_promotion_correct(self):
        """Type promotion produces correct results."""
        x = rp.arange(100000, dtype="float32")
        y = rp.arange(100000, dtype="float64")
        result = x + y
        expected = np.arange(100000, dtype=np.float32) + np.arange(100000, dtype=np.float64)
        assert_eq(result, expected)

    def test_integer_division_correct(self):
        """Integer division produces float result."""
        x = rp.arange(1, 100001, dtype="int64")
        result = x / 2
        expected = np.arange(1, 100001, dtype=np.int64) / 2
        assert_eq(result, expected)


class TestViewHandling:
    """Verify views are handled correctly."""

    def test_view_not_elided(self):
        """Views should not have buffer reused (offset != 0)."""
        x = rp.arange(200000, dtype="float64")
        view = x[100000:]  # View into second half
        result = view + 1 + 2
        expected = np.arange(100000, 200000, dtype=np.float64) + 1 + 2
        assert_eq(result, expected)
        # Original should be unchanged
        expected_orig = np.arange(200000, dtype=np.float64)
        assert_eq(x, expected_orig)

    def test_slice_preserves_original(self):
        """Sliced arrays don't modify original."""
        x = rp.arange(200000, dtype="float64")
        view = x[::2]  # Strided view
        _ = view + 1
        expected = np.arange(200000, dtype=np.float64)
        assert_eq(x, expected)
