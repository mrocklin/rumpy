"""Tests for logical operations.

Parametrizes over operations for concise comprehensive coverage.
See designs/testing.md for testing philosophy.
"""

import numpy as np
import pytest

import rumpy as rp
from conftest import CORE_SHAPES, NUMERIC_DTYPES, SHAPES_BROADCAST, SHAPES_EMPTY
from helpers import assert_eq, make_pair

# === Logical operation categories ===

BINARY_LOGICAL_OPS = ["logical_and", "logical_or", "logical_xor"]
UNARY_LOGICAL_OPS = ["logical_not"]


# === Parametrized tests ===


class TestBinaryLogicalOps:
    """Test binary logical operations: and, or, xor."""

    @pytest.mark.parametrize("op", BINARY_LOGICAL_OPS)
    def test_bool_arrays(self, op):
        """Test with boolean arrays."""
        n1 = np.array([True, True, False, False])
        n2 = np.array([True, False, True, False])
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        rp_fn = getattr(rp, op)
        np_fn = getattr(np, op)
        assert_eq(rp_fn(r1, r2), np_fn(n1, n2))

    @pytest.mark.parametrize("op", BINARY_LOGICAL_OPS)
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_numeric_dtypes(self, op, dtype):
        """Test with numeric arrays (0 is False, nonzero is True)."""
        n1 = np.array([0, 1, 2, 0, 5], dtype=dtype)
        n2 = np.array([0, 0, 1, 3, 0], dtype=dtype)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        rp_fn = getattr(rp, op)
        np_fn = getattr(np, op)
        assert_eq(rp_fn(r1, r2), np_fn(n1, n2))

    @pytest.mark.parametrize("op", BINARY_LOGICAL_OPS)
    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, op, shape):
        """Test various shapes."""
        # Create boolean arrays with alternating pattern
        size = int(np.prod(shape))
        n1 = np.array([i % 2 == 0 for i in range(size)], dtype=bool).reshape(shape)
        n2 = np.array([i % 3 == 0 for i in range(size)], dtype=bool).reshape(shape)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        rp_fn = getattr(rp, op)
        np_fn = getattr(np, op)
        assert_eq(rp_fn(r1, r2), np_fn(n1, n2))

    @pytest.mark.parametrize("op", BINARY_LOGICAL_OPS)
    def test_scalar_bool(self, op):
        """Test with scalar operands (as single-element arrays)."""
        n = np.array([True, False, True, False])
        r = rp.asarray(n)
        rp_fn = getattr(rp, op)
        np_fn = getattr(np, op)
        # Test array op scalar (convert scalar to array for rumpy)
        n_true = np.array([True])
        n_false = np.array([False])
        r_true = rp.asarray(n_true)
        r_false = rp.asarray(n_false)
        assert_eq(rp_fn(r, r_true), np_fn(n, True))
        assert_eq(rp_fn(r, r_false), np_fn(n, False))

    @pytest.mark.parametrize("op", BINARY_LOGICAL_OPS)
    def test_scalar_numeric(self, op):
        """Test with numeric scalars (as single-element arrays)."""
        n = np.array([0, 1, 2, 0, 5], dtype=np.float64)
        r = rp.asarray(n)
        rp_fn = getattr(rp, op)
        np_fn = getattr(np, op)
        # Test array op scalar (convert scalar to array for rumpy)
        n_zero = np.array([0.0])
        n_one = np.array([1.0])
        r_zero = rp.asarray(n_zero)
        r_one = rp.asarray(n_one)
        assert_eq(rp_fn(r, r_zero), np_fn(n, 0.0))
        assert_eq(rp_fn(r, r_one), np_fn(n, 1.0))

    @pytest.mark.parametrize("op", BINARY_LOGICAL_OPS)
    @pytest.mark.parametrize("shapes", SHAPES_BROADCAST)
    def test_broadcast(self, op, shapes):
        """Test broadcasting."""
        shape1, shape2 = shapes
        # Create boolean arrays
        size1 = int(np.prod(shape1))
        size2 = int(np.prod(shape2))
        n1 = np.array([i % 2 == 0 for i in range(size1)], dtype=bool).reshape(shape1)
        n2 = np.array([i % 2 == 1 for i in range(size2)], dtype=bool).reshape(shape2)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        rp_fn = getattr(rp, op)
        np_fn = getattr(np, op)
        assert_eq(rp_fn(r1, r2), np_fn(n1, n2))

    @pytest.mark.parametrize("op", BINARY_LOGICAL_OPS)
    @pytest.mark.parametrize("empty_shape", SHAPES_EMPTY)
    def test_empty(self, op, empty_shape):
        """Test empty arrays."""
        n1 = np.array([], dtype=bool).reshape(empty_shape)
        n2 = np.array([], dtype=bool).reshape(empty_shape)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        rp_fn = getattr(rp, op)
        np_fn = getattr(np, op)
        assert_eq(rp_fn(r1, r2), np_fn(n1, n2))


class TestLogicalNot:
    """Test logical_not (unary operation)."""

    def test_bool_array(self):
        """Test with boolean array."""
        n = np.array([True, False, True, False, True])
        r = rp.asarray(n)
        assert_eq(rp.logical_not(r), np.logical_not(n))

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_numeric_dtypes(self, dtype):
        """Test with numeric arrays (0 is False, nonzero is True)."""
        # Use positive values for unsigned types, include negative for signed
        if "uint" in dtype:
            n = np.array([0, 1, 2, 0, 5, 10], dtype=dtype)
        else:
            n = np.array([0, 1, 2, 0, -5, 10], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(rp.logical_not(r), np.logical_not(n))

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        """Test various shapes."""
        size = int(np.prod(shape))
        n = np.array([i % 2 == 0 for i in range(size)], dtype=bool).reshape(shape)
        r = rp.asarray(n)
        assert_eq(rp.logical_not(r), np.logical_not(n))

    def test_scalar(self):
        """Test with scalar values (as single-element arrays)."""
        # Boolean scalars (convert to arrays for rumpy and numpy)
        assert_eq(rp.logical_not(rp.asarray([True])), np.logical_not(np.array([True])))
        assert_eq(rp.logical_not(rp.asarray([False])), np.logical_not(np.array([False])))
        # Numeric scalars (convert to arrays for rumpy and numpy)
        assert_eq(rp.logical_not(rp.asarray([0])), np.logical_not(np.array([0])))
        assert_eq(rp.logical_not(rp.asarray([1])), np.logical_not(np.array([1])))
        assert_eq(rp.logical_not(rp.asarray([5.0])), np.logical_not(np.array([5.0])))

    @pytest.mark.parametrize("empty_shape", SHAPES_EMPTY)
    def test_empty(self, empty_shape):
        """Test empty arrays."""
        n = np.array([], dtype=bool).reshape(empty_shape)
        r = rp.asarray(n)
        assert_eq(rp.logical_not(r), np.logical_not(n))

    def test_single_element(self):
        """Test single element arrays."""
        # Boolean
        n = np.array([True])
        r = rp.asarray(n)
        assert_eq(rp.logical_not(r), np.logical_not(n))
        # Numeric
        n = np.array([0], dtype=np.int64)
        r = rp.asarray(n)
        assert_eq(rp.logical_not(r), np.logical_not(n))


class TestLogicalTruthTables:
    """Verify truth tables for logical operations."""

    def test_logical_and_truth_table(self):
        """Verify AND truth table."""
        n1 = np.array([True, True, False, False])
        n2 = np.array([True, False, True, False])
        expected = np.array([True, False, False, False])
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        assert_eq(rp.logical_and(r1, r2), expected)

    def test_logical_or_truth_table(self):
        """Verify OR truth table."""
        n1 = np.array([True, True, False, False])
        n2 = np.array([True, False, True, False])
        expected = np.array([True, True, True, False])
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        assert_eq(rp.logical_or(r1, r2), expected)

    def test_logical_xor_truth_table(self):
        """Verify XOR truth table."""
        n1 = np.array([True, True, False, False])
        n2 = np.array([True, False, True, False])
        expected = np.array([False, True, True, False])
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        assert_eq(rp.logical_xor(r1, r2), expected)

    def test_logical_not_truth_table(self):
        """Verify NOT truth table."""
        n = np.array([True, False])
        expected = np.array([False, True])
        r = rp.asarray(n)
        assert_eq(rp.logical_not(r), expected)


class TestLogicalMixedTypes:
    """Test logical operations with mixed numeric and boolean types."""

    def test_bool_and_numeric(self):
        """Test boolean array with numeric array."""
        nb = np.array([True, False, True, False])
        nn = np.array([0, 0, 1, 1], dtype=np.int64)
        rb, rn = rp.asarray(nb), rp.asarray(nn)
        assert_eq(rp.logical_and(rb, rn), np.logical_and(nb, nn))
        assert_eq(rp.logical_or(rb, rn), np.logical_or(nb, nn))
        assert_eq(rp.logical_xor(rb, rn), np.logical_xor(nb, nn))

    def test_float_and_int(self):
        """Test float array with int array."""
        nf = np.array([0.0, 1.0, 2.5, 0.0], dtype=np.float64)
        ni = np.array([0, 0, 1, 2], dtype=np.int64)
        rf, ri = rp.asarray(nf), rp.asarray(ni)
        assert_eq(rp.logical_and(rf, ri), np.logical_and(nf, ni))
        assert_eq(rp.logical_or(rf, ri), np.logical_or(nf, ni))
        assert_eq(rp.logical_xor(rf, ri), np.logical_xor(nf, ni))


class TestLogicalEdgeCases:
    """Test edge cases for logical operations."""

    def test_negative_numbers(self):
        """Negative numbers should be truthy."""
        n = np.array([-5, -1, 0, 1, 5], dtype=np.int64)
        r = rp.asarray(n)
        # All negative numbers are truthy
        assert_eq(rp.logical_not(r), np.logical_not(n))

    def test_very_small_floats(self):
        """Very small (but nonzero) floats should be truthy."""
        n = np.array([1e-100, 0.0, 1e-100], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.logical_not(r), np.logical_not(n))

    def test_all_true(self):
        """Array of all True values."""
        n = np.ones(10, dtype=bool)
        r = rp.asarray(n)
        assert_eq(rp.logical_not(r), np.logical_not(n))
        assert_eq(rp.logical_and(r, r), np.logical_and(n, n))

    def test_all_false(self):
        """Array of all False values."""
        n = np.zeros(10, dtype=bool)
        r = rp.asarray(n)
        assert_eq(rp.logical_not(r), np.logical_not(n))
        assert_eq(rp.logical_or(r, r), np.logical_or(n, n))

    def test_single_true_false(self):
        """Single element True and False."""
        nt = np.array([True])
        nf = np.array([False])
        rt, rf = rp.asarray(nt), rp.asarray(nf)
        assert_eq(rp.logical_and(rt, rf), np.logical_and(nt, nf))
        assert_eq(rp.logical_or(rt, rf), np.logical_or(nt, nf))
        assert_eq(rp.logical_xor(rt, rf), np.logical_xor(nt, nf))


class TestLogicalOutputDtype:
    """Verify that logical operations always return bool dtype."""

    @pytest.mark.parametrize("op", BINARY_LOGICAL_OPS)
    def test_binary_ops_return_bool(self, op):
        """Binary logical ops return bool even with numeric input."""
        n1 = np.array([1, 2, 3], dtype=np.float64)
        n2 = np.array([1, 0, 1], dtype=np.int64)
        r1, r2 = rp.asarray(n1), rp.asarray(n2)
        rp_fn = getattr(rp, op)
        result = rp_fn(r1, r2)
        assert result.dtype == "bool"

    def test_logical_not_returns_bool(self):
        """Logical not returns bool even with numeric input."""
        n = np.array([1, 0, 1, 2], dtype=np.int64)
        r = rp.asarray(n)
        result = rp.logical_not(r)
        assert result.dtype == "bool"
