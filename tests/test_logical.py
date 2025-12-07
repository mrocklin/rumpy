"""Tests for logical operations: logical_and, logical_or, logical_not, logical_xor."""
import numpy as np
import rumpy as rp
from helpers import assert_eq


class TestLogicalAnd:
    """Test rp.logical_and against np.logical_and."""

    def test_basic_bool(self):
        """Basic boolean arrays."""
        a = rp.array([True, True, False, False])
        b = rp.array([True, False, True, False])
        r = rp.logical_and(a, b)
        n = np.logical_and([True, True, False, False], [True, False, True, False])
        assert_eq(r, n)

    def test_int_arrays(self):
        """Integer arrays (nonzero is True)."""
        a = rp.array([1, 2, 0, 5])
        b = rp.array([1, 0, 1, 0])
        r = rp.logical_and(a, b)
        n = np.logical_and([1, 2, 0, 5], [1, 0, 1, 0])
        assert_eq(r, n)

    def test_float_arrays(self):
        """Float arrays (nonzero is True)."""
        a = rp.array([1.0, 0.0, 2.5, 0.0])
        b = rp.array([0.5, 0.0, 0.0, 1.0])
        r = rp.logical_and(a, b)
        n = np.logical_and([1.0, 0.0, 2.5, 0.0], [0.5, 0.0, 0.0, 1.0])
        assert_eq(r, n)

    def test_broadcasting(self):
        """Test broadcasting."""
        a = rp.array([[1, 0], [1, 1]])
        b = rp.array([1, 0])
        r = rp.logical_and(a, b)
        n = np.logical_and([[1, 0], [1, 1]], [1, 0])
        assert_eq(r, n)

    def test_scalar_broadcast(self):
        """Scalar broadcasted against array."""
        a = rp.array([1, 0, 1, 0])
        b = rp.array([1])
        r = rp.logical_and(a, b)
        n = np.logical_and([1, 0, 1, 0], [1])
        assert_eq(r, n)

    def test_empty_arrays(self):
        """Empty arrays."""
        a = rp.array([])
        b = rp.array([])
        r = rp.logical_and(a, b)
        n = np.logical_and([], [])
        assert r.shape == n.shape


class TestLogicalOr:
    """Test rp.logical_or against np.logical_or."""

    def test_basic_bool(self):
        """Basic boolean arrays."""
        a = rp.array([True, True, False, False])
        b = rp.array([True, False, True, False])
        r = rp.logical_or(a, b)
        n = np.logical_or([True, True, False, False], [True, False, True, False])
        assert_eq(r, n)

    def test_int_arrays(self):
        """Integer arrays (nonzero is True)."""
        a = rp.array([1, 2, 0, 0])
        b = rp.array([1, 0, 1, 0])
        r = rp.logical_or(a, b)
        n = np.logical_or([1, 2, 0, 0], [1, 0, 1, 0])
        assert_eq(r, n)

    def test_float_arrays(self):
        """Float arrays (nonzero is True)."""
        a = rp.array([1.0, 0.0, 0.0, 0.0])
        b = rp.array([0.0, 0.0, 0.0, 1.0])
        r = rp.logical_or(a, b)
        n = np.logical_or([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        assert_eq(r, n)

    def test_broadcasting(self):
        """Test broadcasting."""
        a = rp.array([[0, 0], [1, 0]])
        b = rp.array([0, 1])
        r = rp.logical_or(a, b)
        n = np.logical_or([[0, 0], [1, 0]], [0, 1])
        assert_eq(r, n)


class TestLogicalNot:
    """Test rp.logical_not against np.logical_not."""

    def test_basic_bool(self):
        """Basic boolean array."""
        a = rp.array([True, False, True, False])
        r = rp.logical_not(a)
        n = np.logical_not([True, False, True, False])
        assert_eq(r, n)

    def test_int_array(self):
        """Integer array (nonzero is True)."""
        a = rp.array([0, 1, 2, 0, -1])
        r = rp.logical_not(a)
        n = np.logical_not([0, 1, 2, 0, -1])
        assert_eq(r, n)

    def test_float_array(self):
        """Float array (nonzero is True)."""
        a = rp.array([0.0, 1.0, 0.5, 0.0])
        r = rp.logical_not(a)
        n = np.logical_not([0.0, 1.0, 0.5, 0.0])
        assert_eq(r, n)

    def test_2d(self):
        """2D array."""
        a = rp.array([[0, 1], [1, 0]])
        r = rp.logical_not(a)
        n = np.logical_not([[0, 1], [1, 0]])
        assert_eq(r, n)

    def test_empty(self):
        """Empty array."""
        a = rp.array([])
        r = rp.logical_not(a)
        n = np.logical_not([])
        assert r.shape == n.shape


class TestLogicalXor:
    """Test rp.logical_xor against np.logical_xor."""

    def test_basic_bool(self):
        """Basic boolean arrays."""
        a = rp.array([True, True, False, False])
        b = rp.array([True, False, True, False])
        r = rp.logical_xor(a, b)
        n = np.logical_xor([True, True, False, False], [True, False, True, False])
        assert_eq(r, n)

    def test_int_arrays(self):
        """Integer arrays (nonzero is True)."""
        a = rp.array([1, 1, 0, 0])
        b = rp.array([1, 0, 1, 0])
        r = rp.logical_xor(a, b)
        n = np.logical_xor([1, 1, 0, 0], [1, 0, 1, 0])
        assert_eq(r, n)

    def test_float_arrays(self):
        """Float arrays (nonzero is True)."""
        a = rp.array([1.0, 1.0, 0.0, 0.0])
        b = rp.array([1.0, 0.0, 1.0, 0.0])
        r = rp.logical_xor(a, b)
        n = np.logical_xor([1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0])
        assert_eq(r, n)

    def test_broadcasting(self):
        """Test broadcasting."""
        a = rp.array([[1, 0], [0, 1]])
        b = rp.array([1, 1])
        r = rp.logical_xor(a, b)
        n = np.logical_xor([[1, 0], [0, 1]], [1, 1])
        assert_eq(r, n)


class TestOutputDtype:
    """Verify that logical operations return bool dtype."""

    def test_logical_and_dtype(self):
        a = rp.array([1, 2, 3])
        b = rp.array([1, 0, 1])
        r = rp.logical_and(a, b)
        assert r.dtype == "bool"

    def test_logical_or_dtype(self):
        a = rp.array([1.0, 2.0, 3.0])
        b = rp.array([1.0, 0.0, 1.0])
        r = rp.logical_or(a, b)
        assert r.dtype == "bool"

    def test_logical_not_dtype(self):
        a = rp.array([1, 0, 1])
        r = rp.logical_not(a)
        assert r.dtype == "bool"

    def test_logical_xor_dtype(self):
        a = rp.array([True, False])
        b = rp.array([False, True])
        r = rp.logical_xor(a, b)
        assert r.dtype == "bool"
