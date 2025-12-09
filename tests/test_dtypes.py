"""Tests for dtype system - new dtypes and parametric types."""

import numpy as np
import pytest

import rumpy as rp
from helpers import assert_eq


class TestUnsignedIntegers:
    """Test unsigned integer dtypes."""

    def test_uint8_zeros(self):
        r = rp.zeros(10, dtype="uint8")
        n = np.zeros(10, dtype=np.uint8)
        assert_eq(r, n)

    def test_uint8_ones(self):
        r = rp.ones(10, dtype="uint8")
        n = np.ones(10, dtype=np.uint8)
        assert_eq(r, n)

    def test_uint8_arange(self):
        r = rp.arange(10, dtype="uint8")
        n = np.arange(10, dtype=np.uint8)
        assert_eq(r, n)

    def test_uint8_itemsize(self):
        r = rp.zeros(10, dtype="uint8")
        assert r.itemsize == 1

    def test_uint32_zeros(self):
        r = rp.zeros(10, dtype="uint32")
        n = np.zeros(10, dtype=np.uint32)
        assert_eq(r, n)

    def test_uint32_ones(self):
        r = rp.ones(10, dtype="uint32")
        n = np.ones(10, dtype=np.uint32)
        assert_eq(r, n)

    def test_uint32_arange(self):
        r = rp.arange(100, dtype="uint32")
        n = np.arange(100, dtype=np.uint32)
        assert_eq(r, n)

    def test_uint32_itemsize(self):
        r = rp.zeros(10, dtype="uint32")
        assert r.itemsize == 4

    def test_uint64_zeros(self):
        r = rp.zeros(10, dtype="uint64")
        n = np.zeros(10, dtype=np.uint64)
        assert_eq(r, n)

    def test_uint64_ones(self):
        r = rp.ones(10, dtype="uint64")
        n = np.ones(10, dtype=np.uint64)
        assert_eq(r, n)

    def test_uint64_arange(self):
        r = rp.arange(100, dtype="uint64")
        n = np.arange(100, dtype=np.uint64)
        assert_eq(r, n)

    def test_uint64_itemsize(self):
        r = rp.zeros(10, dtype="uint64")
        assert r.itemsize == 8


class TestSmallIntegerDtypes:
    """Test int8, int16, uint16 dtypes."""

    def test_int8_from_numpy(self):
        n = np.array([1, 2, 3], dtype=np.int8)
        r = rp.asarray(n)
        assert_eq(r, n)

    def test_int16_from_numpy(self):
        n = np.array([1, 2, 3], dtype=np.int16)
        r = rp.asarray(n)
        assert_eq(r, n)

    def test_uint16_from_numpy(self):
        n = np.array([1, 2, 3], dtype=np.uint16)
        r = rp.asarray(n)
        assert_eq(r, n)

    def test_int8_creation(self):
        r = rp.zeros(5, dtype="int8")
        n = np.zeros(5, dtype=np.int8)
        assert_eq(r, n)

    def test_int16_creation(self):
        r = rp.zeros(5, dtype="int16")
        n = np.zeros(5, dtype=np.int16)
        assert_eq(r, n)

    def test_uint16_creation(self):
        r = rp.zeros(5, dtype="uint16")
        n = np.zeros(5, dtype=np.uint16)
        assert_eq(r, n)


class TestFloat16Interop:
    """Test float16 numpy interop."""

    def test_float16_from_numpy(self):
        n = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        r = rp.asarray(n)
        assert_eq(r, n)

    def test_float16_creation(self):
        r = rp.zeros(5, dtype="float16")
        n = np.zeros(5, dtype=np.float16)
        assert_eq(r, n)


class TestUnsignedInterop:
    """Test numpy interop with unsigned types."""

    def test_numpy_uint8_to_rumpy(self):
        n = np.array([1, 2, 3, 255], dtype=np.uint8)
        r = rp.asarray(n)
        assert_eq(r, n)

    def test_numpy_uint32_to_rumpy(self):
        n = np.arange(10, dtype=np.uint32)
        r = rp.asarray(n)
        assert_eq(r, n)

    def test_numpy_uint64_to_rumpy(self):
        n = np.arange(10, dtype=np.uint64)
        r = rp.asarray(n)
        assert_eq(r, n)

    def test_rumpy_uint8_to_numpy(self):
        r = rp.arange(10, dtype="uint8")
        n = np.asarray(r)
        assert n.dtype == np.uint8
        assert_eq(r, n)

    def test_rumpy_uint32_to_numpy(self):
        r = rp.arange(10, dtype="uint32")
        n = np.asarray(r)
        assert n.dtype == np.uint32
        assert_eq(r, n)

    def test_rumpy_uint64_to_numpy(self):
        r = rp.arange(10, dtype="uint64")
        n = np.asarray(r)
        assert n.dtype == np.uint64
        assert_eq(r, n)


class TestUnsignedOperations:
    """Test operations on unsigned types."""

    def test_uint8_add(self):
        r1 = rp.arange(5, dtype="uint8")
        r2 = rp.ones(5, dtype="uint8")
        n1 = np.arange(5, dtype=np.uint8)
        n2 = np.ones(5, dtype=np.uint8)
        assert_eq(r1 + r2, n1 + n2)

    def test_uint32_mul(self):
        r = rp.arange(5, dtype="uint32")
        n = np.arange(5, dtype=np.uint32)
        assert_eq(r * r, n * n)

    def test_uint64_sum(self):
        r = rp.arange(10, dtype="uint64")
        n = np.arange(10, dtype=np.uint64)
        assert r.sum() == n.sum()

    def test_uint8_2d(self):
        r = rp.ones((3, 4), dtype="uint8")
        n = np.ones((3, 4), dtype=np.uint8)
        assert_eq(r, n)

    def test_uint32_reshape(self):
        r = rp.arange(12, dtype="uint32").reshape(3, 4)
        n = np.arange(12, dtype=np.uint32).reshape(3, 4)
        assert_eq(r, n)


class TestDateTime64:
    """Test datetime64 parametric dtype."""

    def test_datetime64_ns_zeros(self):
        r = rp.zeros(5, dtype="datetime64[ns]")
        n = np.zeros(5, dtype="datetime64[ns]")
        assert r.dtype == "datetime64[ns]"
        assert r.itemsize == 8
        # NumPy will interpret our array as datetime64 via __array_interface__
        assert_eq(r, n)

    def test_datetime64_us_zeros(self):
        r = rp.zeros(5, dtype="datetime64[us]")
        n = np.zeros(5, dtype="datetime64[us]")
        assert r.dtype == "datetime64[us]"
        assert_eq(r, n)

    def test_datetime64_ms_zeros(self):
        r = rp.zeros(5, dtype="datetime64[ms]")
        n = np.zeros(5, dtype="datetime64[ms]")
        assert r.dtype == "datetime64[ms]"
        assert_eq(r, n)

    def test_datetime64_s_zeros(self):
        r = rp.zeros(5, dtype="datetime64[s]")
        n = np.zeros(5, dtype="datetime64[s]")
        assert r.dtype == "datetime64[s]"
        assert_eq(r, n)

    def test_datetime64_different_units_not_equal(self):
        """Different time units should be different dtypes."""
        r_ns = rp.zeros(1, dtype="datetime64[ns]")
        r_us = rp.zeros(1, dtype="datetime64[us]")
        assert r_ns.dtype != r_us.dtype

    def test_datetime64_same_unit_equal(self):
        """Same time unit should be equal dtype."""
        r1 = rp.zeros(1, dtype="datetime64[ns]")
        r2 = rp.ones(1, dtype="datetime64[ns]")
        assert r1.dtype == r2.dtype


class TestDateTime64Interop:
    """Test numpy interop with datetime64."""

    def test_numpy_datetime64_ns_to_rumpy(self):
        n = np.array(['2024-01-01', '2024-01-02'], dtype="datetime64[ns]")
        r = rp.asarray(n)
        assert r.dtype == "datetime64[ns]"
        # Values should match when converted back
        assert_eq(r, n)

    def test_numpy_datetime64_s_to_rumpy(self):
        n = np.array(['2024-01-01', '2024-01-02'], dtype="datetime64[s]")
        r = rp.asarray(n)
        assert r.dtype == "datetime64[s]"
        assert_eq(r, n)

    def test_rumpy_datetime64_to_numpy(self):
        """Round-trip: rumpy -> numpy -> compare."""
        r = rp.zeros(3, dtype="datetime64[ns]")
        n = np.asarray(r)
        # NumPy should see it as datetime64[ns]
        assert n.dtype == np.dtype("datetime64[ns]")

    def test_datetime64_2d(self):
        r = rp.zeros((2, 3), dtype="datetime64[ns]")
        n = np.zeros((2, 3), dtype="datetime64[ns]")
        assert r.shape == (2, 3)
        assert_eq(r, n)


class TestComplex128:
    """Test complex128 dtype."""

    def test_complex128_zeros(self):
        r = rp.zeros(5, dtype="complex128")
        n = np.zeros(5, dtype=np.complex128)
        assert r.dtype == "complex128"
        assert r.itemsize == 16
        assert_eq(r, n)

    def test_complex128_ones(self):
        r = rp.ones(5, dtype="complex128")
        n = np.ones(5, dtype=np.complex128)
        assert_eq(r, n)

    def test_complex128_2d(self):
        r = rp.zeros((3, 4), dtype="complex128")
        n = np.zeros((3, 4), dtype=np.complex128)
        assert_eq(r, n)


class TestComplex128Interop:
    """Test numpy interop with complex128."""

    def test_numpy_complex128_to_rumpy(self):
        n = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex128)
        r = rp.asarray(n)
        assert r.dtype == "complex128"
        assert_eq(r, n)

    def test_rumpy_complex128_to_numpy(self):
        r = rp.zeros(3, dtype="complex128")
        n = np.asarray(r)
        assert n.dtype == np.complex128
        assert_eq(r, n)


class TestDtypeInference:
    """Test dtype inference from Python list values."""

    def test_int_list_infers_int64(self):
        r = rp.array([1, 2, 3])
        n = np.array([1, 2, 3])
        assert r.dtype == str(n.dtype)

    def test_float_list_infers_float64(self):
        r = rp.array([1.0, 2.0, 3.0])
        n = np.array([1.0, 2.0, 3.0])
        assert r.dtype == str(n.dtype)

    def test_bool_list_infers_bool(self):
        r = rp.array([True, False, True])
        assert r.dtype == "bool"

    def test_nested_int_list_infers_int64(self):
        r = rp.array([[1, 2], [3, 4]])
        n = np.array([[1, 2], [3, 4]])
        assert r.dtype == str(n.dtype)

    def test_explicit_dtype_overrides_inference(self):
        """Explicit dtype should override inference."""
        r = rp.array([1, 2, 3], dtype="float32")
        assert r.dtype == "float32"


class TestBitwiseDtypePreservation:
    """Test that bitwise operations preserve dtype."""

    def test_bitwise_and_preserves_uint8(self):
        a = rp.array([1, 2, 3], dtype="uint8")
        b = rp.array([1, 1, 1], dtype="uint8")
        r = a & b
        n = np.array([1, 2, 3], dtype=np.uint8) & np.array([1, 1, 1], dtype=np.uint8)
        assert r.dtype == "uint8"
        assert_eq(r, n)

    def test_bitwise_or_preserves_int32(self):
        a = rp.array([1, 2, 3], dtype="int32")
        b = rp.array([4, 5, 6], dtype="int32")
        r = a | b
        n = np.array([1, 2, 3], dtype=np.int32) | np.array([4, 5, 6], dtype=np.int32)
        assert r.dtype == "int32"
        assert_eq(r, n)

    def test_bitwise_xor_preserves_uint64(self):
        a = rp.array([1, 2, 3], dtype="uint64")
        b = rp.array([3, 2, 1], dtype="uint64")
        r = a ^ b
        n = np.array([1, 2, 3], dtype=np.uint64) ^ np.array([3, 2, 1], dtype=np.uint64)
        assert r.dtype == "uint64"
        assert_eq(r, n)

    def test_bitwise_not_preserves_int64(self):
        a = rp.array([1, 2, 3], dtype="int64")
        r = ~a
        n = ~np.array([1, 2, 3], dtype=np.int64)
        assert r.dtype == "int64"
        assert_eq(r, n)

    def test_bitwise_not_preserves_bool(self):
        a = rp.array([True, False, True])
        r = ~a
        n = ~np.array([True, False, True])
        assert r.dtype == "bool"
        assert_eq(r, n)

    def test_left_shift_preserves_dtype(self):
        a = rp.array([1, 2, 4], dtype="int32")
        b = rp.array([1, 1, 1], dtype="int32")
        r = a << b
        n = np.array([1, 2, 4], dtype=np.int32) << np.array([1, 1, 1], dtype=np.int32)
        assert r.dtype == "int32"
        assert_eq(r, n)

    def test_right_shift_preserves_dtype(self):
        a = rp.array([8, 4, 2], dtype="uint16")
        b = rp.array([1, 1, 1], dtype="uint16")
        r = a >> b
        n = np.array([8, 4, 2], dtype=np.uint16) >> np.array([1, 1, 1], dtype=np.uint16)
        assert r.dtype == "uint16"
        assert_eq(r, n)
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
