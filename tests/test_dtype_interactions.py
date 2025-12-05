"""Tests for dtype interaction behavior - comparing against NumPy."""

import numpy as np
import pytest

import rumpy as rp
from helpers import assert_eq


class TestNumericPromotion:
    """Test type promotion rules for numeric types."""

    def test_int64_plus_float32(self):
        """int64 + float32 -> float64 in NumPy."""
        n1 = np.arange(3, dtype="int64")
        n2 = np.ones(3, dtype="float32")
        r1 = rp.arange(3, dtype="int64")
        r2 = rp.ones(3, dtype="float32")

        n_result = n1 + n2
        r_result = r1 + r2

        assert r_result.dtype == n_result.dtype.name
        assert_eq(r_result, n_result)

    def test_int64_plus_float64(self):
        """int64 + float64 -> float64."""
        n1 = np.arange(3, dtype="int64")
        n2 = np.ones(3, dtype="float64")
        r1 = rp.arange(3, dtype="int64")
        r2 = rp.ones(3, dtype="float64")

        n_result = n1 + n2
        r_result = r1 + r2

        assert r_result.dtype == n_result.dtype.name
        assert_eq(r_result, n_result)

    def test_int32_plus_float64(self):
        """int32 + float64 -> float64."""
        n1 = np.arange(3, dtype="int32")
        n2 = np.ones(3, dtype="float64")
        r1 = rp.arange(3, dtype="int32")
        r2 = rp.ones(3, dtype="float64")

        n_result = n1 + n2
        r_result = r1 + r2

        assert r_result.dtype == n_result.dtype.name
        assert_eq(r_result, n_result)

    def test_float32_plus_float64(self):
        """float32 + float64 -> float64."""
        n1 = np.ones(3, dtype="float32")
        n2 = np.ones(3, dtype="float64")
        r1 = rp.ones(3, dtype="float32")
        r2 = rp.ones(3, dtype="float64")

        n_result = n1 + n2
        r_result = r1 + r2

        assert r_result.dtype == n_result.dtype.name
        assert_eq(r_result, n_result)

    def test_uint8_plus_int64(self):
        """uint8 + int64 -> int64."""
        n1 = np.arange(3, dtype="uint8")
        n2 = np.arange(3, dtype="int64")
        r1 = rp.arange(3, dtype="uint8")
        r2 = rp.arange(3, dtype="int64")

        n_result = n1 + n2
        r_result = r1 + r2

        assert r_result.dtype == n_result.dtype.name
        assert_eq(r_result, n_result)

    def test_uint8_plus_float32(self):
        """uint8 + float32 -> float32."""
        n1 = np.arange(3, dtype="uint8")
        n2 = np.ones(3, dtype="float32")
        r1 = rp.arange(3, dtype="uint8")
        r2 = rp.ones(3, dtype="float32")

        n_result = n1 + n2
        r_result = r1 + r2

        assert r_result.dtype == n_result.dtype.name
        assert_eq(r_result, n_result)

    def test_bool_plus_int64(self):
        """bool + int64 -> int64."""
        n1 = np.array([True, False, True], dtype="bool")
        n2 = np.arange(3, dtype="int64")
        r1 = rp.asarray(np.array([True, False, True], dtype="bool"))
        r2 = rp.arange(3, dtype="int64")

        n_result = n1 + n2
        r_result = r1 + r2

        assert r_result.dtype == n_result.dtype.name
        assert_eq(r_result, n_result)

    def test_bool_plus_float64(self):
        """bool + float64 -> float64."""
        n1 = np.array([True, False, True], dtype="bool")
        n2 = np.ones(3, dtype="float64")
        r1 = rp.asarray(np.array([True, False, True], dtype="bool"))
        r2 = rp.ones(3, dtype="float64")

        n_result = n1 + n2
        r_result = r1 + r2

        assert r_result.dtype == n_result.dtype.name
        assert_eq(r_result, n_result)


class TestBoolOperations:
    """Test bool-bool operations."""

    def test_bool_plus_bool(self):
        """bool + bool -> bool in NumPy."""
        n1 = np.array([True, False], dtype="bool")
        n2 = np.array([True, True], dtype="bool")
        r1 = rp.asarray(n1)
        r2 = rp.asarray(n2)

        n_result = n1 + n2
        r_result = r1 + r2

        assert r_result.dtype == n_result.dtype.name
        assert_eq(r_result, n_result)

    def test_bool_mul_bool(self):
        """bool * bool -> bool in NumPy."""
        n1 = np.array([True, False], dtype="bool")
        n2 = np.array([True, True], dtype="bool")
        r1 = rp.asarray(n1)
        r2 = rp.asarray(n2)

        n_result = n1 * n2
        r_result = r1 * r2

        assert r_result.dtype == n_result.dtype.name
        assert_eq(r_result, n_result)


class TestComplexPromotion:
    """Test complex type promotion."""

    def test_complex128_plus_float64(self):
        """complex128 + float64 -> complex128."""
        n1 = np.array([1+2j, 3+4j], dtype="complex128")
        n2 = np.array([1.0, 2.0], dtype="float64")
        r1 = rp.asarray(n1)
        r2 = rp.asarray(n2)

        n_result = n1 + n2
        r_result = r1 + r2

        assert r_result.dtype == n_result.dtype.name
        assert_eq(r_result, n_result)

    def test_complex128_plus_int64(self):
        """complex128 + int64 -> complex128."""
        n1 = np.array([1+2j, 3+4j], dtype="complex128")
        n2 = np.array([1, 2], dtype="int64")
        r1 = rp.asarray(n1)
        r2 = rp.asarray(n2)

        n_result = n1 + n2
        r_result = r1 + r2

        assert r_result.dtype == n_result.dtype.name
        assert_eq(r_result, n_result)

    def test_complex128_mul_bool(self):
        """complex128 * bool -> complex128."""
        n1 = np.array([1+2j, 3+4j], dtype="complex128")
        n2 = np.array([True, False], dtype="bool")
        r1 = rp.asarray(n1)
        r2 = rp.asarray(n2)

        n_result = n1 * n2
        r_result = r1 * r2

        assert r_result.dtype == n_result.dtype.name
        assert_eq(r_result, n_result)


class TestDatetimeOperations:
    """Test datetime operations - many should error like NumPy."""

    def test_datetime_add_datetime_should_error(self):
        """datetime64 + datetime64 -> error in NumPy."""
        n1 = np.array(['2024-01-01'], dtype='datetime64[ns]')
        n2 = np.array(['2024-01-02'], dtype='datetime64[ns]')
        r1 = rp.asarray(n1)
        r2 = rp.asarray(n2)

        # NumPy raises UFuncTypeError
        with pytest.raises(TypeError):
            _ = n1 + n2

        # Rumpy should also error (or return None which raises in Python)
        with pytest.raises((TypeError, AttributeError)):
            _ = r1 + r2

    def test_datetime_sub_datetime(self):
        """datetime64 - datetime64 -> timedelta64 in NumPy."""
        n1 = np.array(['2024-01-02'], dtype='datetime64[ns]')
        n2 = np.array(['2024-01-01'], dtype='datetime64[ns]')

        n_result = n1 - n2
        # Result is timedelta64[ns]
        assert n_result.dtype == np.dtype('timedelta64[ns]')

        # TODO: Once we support timedelta64, test rumpy here
        # For now, just document NumPy's behavior

    def test_datetime_mul_int_should_error(self):
        """datetime64 * int -> error in NumPy."""
        n1 = np.array(['2024-01-01'], dtype='datetime64[ns]')
        n2 = np.array([2], dtype='int64')

        with pytest.raises(TypeError):
            _ = n1 * n2

    def test_datetime_different_units(self):
        """datetime64[ns] + datetime64[ms] -> error in NumPy."""
        n1 = np.array(['2024-01-01'], dtype='datetime64[ns]')
        n2 = np.array(['2024-01-02'], dtype='datetime64[ms]')

        # NumPy raises UFuncTypeError for adding datetimes
        with pytest.raises(TypeError):
            _ = n1 + n2


class TestSameTypeOperations:
    """Test that same-type operations preserve dtype."""

    @pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
    def test_same_type_add(self, dtype):
        """Same type + same type -> same type."""
        n1 = np.arange(5, dtype=dtype)
        n2 = np.ones(5, dtype=dtype)
        r1 = rp.arange(5, dtype=dtype)
        r2 = rp.ones(5, dtype=dtype)

        n_result = n1 + n2
        r_result = r1 + r2

        assert r_result.dtype == dtype
        assert n_result.dtype == np.dtype(dtype)
        assert_eq(r_result, n_result)

    @pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
    def test_same_type_mul(self, dtype):
        """Same type * same type -> same type."""
        n1 = np.arange(5, dtype=dtype)
        n2 = np.arange(5, dtype=dtype)
        r1 = rp.arange(5, dtype=dtype)
        r2 = rp.arange(5, dtype=dtype)

        n_result = n1 * n2
        r_result = r1 * r2

        assert r_result.dtype == dtype
        assert n_result.dtype == np.dtype(dtype)
        assert_eq(r_result, n_result)


class TestUnsignedPromotion:
    """Test unsigned integer promotion."""

    def test_uint8_plus_uint32(self):
        """uint8 + uint32 -> uint32."""
        n1 = np.arange(3, dtype="uint8")
        n2 = np.arange(3, dtype="uint32")
        r1 = rp.arange(3, dtype="uint8")
        r2 = rp.arange(3, dtype="uint32")

        n_result = n1 + n2
        r_result = r1 + r2

        assert r_result.dtype == n_result.dtype.name
        assert_eq(r_result, n_result)

    def test_uint32_plus_uint64(self):
        """uint32 + uint64 -> uint64."""
        n1 = np.arange(3, dtype="uint32")
        n2 = np.arange(3, dtype="uint64")
        r1 = rp.arange(3, dtype="uint32")
        r2 = rp.arange(3, dtype="uint64")

        n_result = n1 + n2
        r_result = r1 + r2

        assert r_result.dtype == n_result.dtype.name
        assert_eq(r_result, n_result)

    def test_uint64_plus_float64(self):
        """uint64 + float64 -> float64."""
        n1 = np.arange(3, dtype="uint64")
        n2 = np.ones(3, dtype="float64")
        r1 = rp.arange(3, dtype="uint64")
        r2 = rp.ones(3, dtype="float64")

        n_result = n1 + n2
        r_result = r1 + r2

        assert r_result.dtype == n_result.dtype.name
        assert_eq(r_result, n_result)
