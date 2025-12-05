"""Tests for new unary operations - floor, ceil, inverse trig."""

import numpy as np
import pytest

import rumpy as rp
from helpers import assert_eq


class TestFloorCeil:
    """Test floor and ceil operations."""

    def test_floor_float64(self):
        n = np.array([1.1, 2.5, 3.9, -1.1, -2.5])
        r = rp.asarray(n)
        assert_eq(rp.floor(r), np.floor(n))

    def test_ceil_float64(self):
        n = np.array([1.1, 2.5, 3.9, -1.1, -2.5])
        r = rp.asarray(n)
        assert_eq(rp.ceil(r), np.ceil(n))

    def test_floor_float32(self):
        n = np.array([1.1, 2.5, 3.9], dtype="float32")
        r = rp.asarray(n)
        result = rp.floor(r)
        assert result.dtype == "float32"
        assert_eq(result, np.floor(n))

    def test_ceil_float32(self):
        n = np.array([1.1, 2.5, 3.9], dtype="float32")
        r = rp.asarray(n)
        result = rp.ceil(r)
        assert result.dtype == "float32"
        assert_eq(result, np.ceil(n))

    def test_floor_integers_noop(self):
        """Floor of integers should be unchanged."""
        n = np.array([1, 2, 3], dtype="int64")
        r = rp.asarray(n)
        # NumPy returns float for floor of int, but we can match the values
        n_result = np.floor(n)
        r_result = rp.floor(r)
        assert_eq(r_result, n_result)

    def test_ceil_integers_noop(self):
        """Ceil of integers should be unchanged."""
        n = np.array([1, 2, 3], dtype="int64")
        r = rp.asarray(n)
        n_result = np.ceil(n)
        r_result = rp.ceil(r)
        assert_eq(r_result, n_result)


class TestInverseTrig:
    """Test inverse trigonometric functions."""

    def test_arcsin(self):
        n = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        r = rp.asarray(n)
        assert_eq(rp.arcsin(r), np.arcsin(n))

    def test_arccos(self):
        n = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        r = rp.asarray(n)
        assert_eq(rp.arccos(r), np.arccos(n))

    def test_arctan(self):
        n = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        r = rp.asarray(n)
        assert_eq(rp.arctan(r), np.arctan(n))

    def test_arcsin_float32(self):
        n = np.array([-1.0, 0.0, 1.0], dtype="float32")
        r = rp.asarray(n)
        result = rp.arcsin(r)
        assert result.dtype == "float32"
        assert_eq(result, np.arcsin(n))

    def test_arccos_float32(self):
        n = np.array([-1.0, 0.0, 1.0], dtype="float32")
        r = rp.asarray(n)
        result = rp.arccos(r)
        assert result.dtype == "float32"
        assert_eq(result, np.arccos(n))

    def test_arctan_float32(self):
        n = np.array([-1.0, 0.0, 1.0], dtype="float32")
        r = rp.asarray(n)
        result = rp.arctan(r)
        assert result.dtype == "float32"
        assert_eq(result, np.arctan(n))

    def test_arcsin_out_of_domain(self):
        """arcsin of value outside [-1, 1] should return nan."""
        n = np.array([2.0, -2.0])
        r = rp.asarray(n)
        n_result = np.arcsin(n)
        r_result = rp.arcsin(r)
        # Both should be nan
        assert np.isnan(n_result).all()
        assert_eq(r_result, n_result)

    def test_arccos_out_of_domain(self):
        """arccos of value outside [-1, 1] should return nan."""
        n = np.array([2.0, -2.0])
        r = rp.asarray(n)
        n_result = np.arccos(n)
        r_result = rp.arccos(r)
        assert np.isnan(n_result).all()
        assert_eq(r_result, n_result)


class TestComplexUnsupported:
    """Test that operations unsupported on complex raise TypeError like NumPy."""

    def test_floor_complex_raises(self):
        """floor on complex128 should raise TypeError."""
        n = np.array([1.5 + 2.5j])
        r = rp.asarray(n)

        with pytest.raises(TypeError):
            np.floor(n)

        with pytest.raises(TypeError):
            rp.floor(r)

    def test_ceil_complex_raises(self):
        """ceil on complex128 should raise TypeError."""
        n = np.array([1.5 + 2.5j])
        r = rp.asarray(n)

        with pytest.raises(TypeError):
            np.ceil(n)

        with pytest.raises(TypeError):
            rp.ceil(r)


class TestComplexInverseTrig:
    """Test inverse trig functions on complex numbers."""

    def test_arcsin_complex(self):
        """arcsin on complex128 should match NumPy."""
        n = np.array([1+2j, 0.5+0.5j, 1+0j, -1+0j, 0+1j])
        r = rp.asarray(n)
        assert_eq(rp.arcsin(r), np.arcsin(n))

    def test_arccos_complex(self):
        """arccos on complex128 should match NumPy."""
        n = np.array([1+2j, 0.5+0.5j, 1+0j, -1+0j, 0+1j])
        r = rp.asarray(n)
        assert_eq(rp.arccos(r), np.arccos(n))

    def test_arctan_complex(self):
        """arctan on complex128 should match NumPy."""
        n = np.array([1+2j, 0.5+0.5j, 1+0j, -1+0j])
        r = rp.asarray(n)
        assert_eq(rp.arctan(r), np.arctan(n))
