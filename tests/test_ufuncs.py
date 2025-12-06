"""Tests for math ufuncs (unary and binary element-wise functions)."""

import numpy as np
import pytest

import rumpy as rp
from helpers import assert_eq


class TestSqrt:
    """Test sqrt ufunc."""

    def test_sqrt_1d(self):
        r = rp.sqrt(rp.arange(0, 10, dtype="float64"))
        n = np.sqrt(np.arange(10, dtype=np.float64))
        assert_eq(r, n)

    def test_sqrt_2d(self):
        r = rp.sqrt(rp.arange(1, 13, dtype="float64").reshape(3, 4))
        n = np.sqrt(np.arange(1, 13, dtype=np.float64).reshape(3, 4))
        assert_eq(r, n)


class TestExp:
    """Test exp ufunc."""

    def test_exp_1d(self):
        r = rp.exp(rp.arange(0, 5, dtype="float64"))
        n = np.exp(np.arange(5, dtype=np.float64))
        assert_eq(r, n)

    def test_exp_negative(self):
        r = rp.exp(rp.arange(-3, 3, dtype="float64"))
        n = np.exp(np.arange(-3, 3, dtype=np.float64))
        assert_eq(r, n)

    def test_exp_2d(self):
        r = rp.exp(rp.arange(6, dtype="float64").reshape(2, 3))
        n = np.exp(np.arange(6, dtype=np.float64).reshape(2, 3))
        assert_eq(r, n)


class TestLog:
    """Test log (natural logarithm) ufunc."""

    def test_log_1d(self):
        r = rp.log(rp.arange(1, 10, dtype="float64"))
        n = np.log(np.arange(1, 10, dtype=np.float64))
        assert_eq(r, n)

    def test_log_2d(self):
        r = rp.log(rp.arange(1, 7, dtype="float64").reshape(2, 3))
        n = np.log(np.arange(1, 7, dtype=np.float64).reshape(2, 3))
        assert_eq(r, n)

    def test_log_exp_inverse(self):
        x = rp.arange(1, 5, dtype="float64")
        assert_eq(rp.exp(rp.log(x)), np.arange(1, 5, dtype=np.float64))


class TestSin:
    """Test sin ufunc."""

    def test_sin_1d(self):
        r = rp.sin(rp.linspace(0, 6.28, 10))
        n = np.sin(np.linspace(0, 6.28, 10))
        assert_eq(r, n)

    def test_sin_2d(self):
        r = rp.sin(rp.linspace(0, 3.14, 6).reshape(2, 3))
        n = np.sin(np.linspace(0, 3.14, 6).reshape(2, 3))
        assert_eq(r, n)


class TestCos:
    """Test cos ufunc."""

    def test_cos_1d(self):
        r = rp.cos(rp.linspace(0, 6.28, 10))
        n = np.cos(np.linspace(0, 6.28, 10))
        assert_eq(r, n)

    def test_cos_2d(self):
        r = rp.cos(rp.linspace(0, 3.14, 6).reshape(2, 3))
        n = np.cos(np.linspace(0, 3.14, 6).reshape(2, 3))
        assert_eq(r, n)


class TestTan:
    """Test tan ufunc."""

    def test_tan_1d(self):
        # Avoid pi/2 where tan is undefined
        r = rp.tan(rp.linspace(0, 1, 10))
        n = np.tan(np.linspace(0, 1, 10))
        assert_eq(r, n)

    def test_tan_2d(self):
        r = rp.tan(rp.linspace(0, 1, 6).reshape(2, 3))
        n = np.tan(np.linspace(0, 1, 6).reshape(2, 3))
        assert_eq(r, n)


class TestSinCosPythagorean:
    """Test sin^2 + cos^2 = 1."""

    def test_pythagorean_identity(self):
        x = rp.linspace(0, 6.28, 100)
        s = rp.sin(x)
        c = rp.cos(x)
        result = s * s + c * c
        expected = rp.ones(100)
        assert_eq(result, np.asarray(expected))


class TestTranscendentalPromotion:
    """Test that transcendentals return same dtype and values as NumPy."""

    @pytest.mark.parametrize(
        "dtype",
        [
            "int64", "int32", "int16",
            "uint64", "uint32", "uint16", "uint8",
            "float64", "float32", "float16",
            "complex128", "complex64",
        ],
    )
    @pytest.mark.parametrize("op", ["exp", "log", "sqrt", "sin", "cos", "tan"])
    def test_transcendental_dtype_and_values(self, dtype, op):
        """Transcendentals should return same dtype and values as NumPy."""
        rp_op = getattr(rp, op)
        np_op = getattr(np, op)

        r_arr = rp.asarray([1, 2, 4], dtype=dtype)
        n_arr = np.array([1, 2, 4], dtype=dtype)

        r_result = rp_op(r_arr)
        n_result = np_op(n_arr)

        # Verify dtypes match exactly
        assert r_result.dtype == str(n_result.dtype), f"Expected {n_result.dtype}, got {r_result.dtype}"
        # Compare values
        assert_eq(r_result, n_result)


ALL_DTYPES = [
    "int64", "int32", "int16",
    "uint64", "uint32", "uint16", "uint8",
    "float64", "float32", "float16",
    "complex128", "complex64",
]


class TestBinaryDtypePromotion:
    """Test binary ops return same dtype and values as NumPy across dtype pairs."""

    @pytest.mark.parametrize("dtype_a", ALL_DTYPES)
    @pytest.mark.parametrize("dtype_b", ALL_DTYPES)
    def test_div_dtype_and_values(self, dtype_a, dtype_b):
        """Division should return same dtype and values as NumPy."""
        r_a = rp.asarray([6, 12, 24], dtype=dtype_a)
        r_b = rp.asarray([2, 3, 4], dtype=dtype_b)
        n_a = np.array([6, 12, 24], dtype=dtype_a)
        n_b = np.array([2, 3, 4], dtype=dtype_b)

        r_result = r_a / r_b
        n_result = n_a / n_b

        assert r_result.dtype == str(n_result.dtype), f"{dtype_a}/{dtype_b}: Expected {n_result.dtype}, got {r_result.dtype}"
        assert_eq(r_result, n_result)

    @pytest.mark.parametrize("dtype_a", ALL_DTYPES)
    @pytest.mark.parametrize("dtype_b", ALL_DTYPES)
    def test_add_dtype_and_values(self, dtype_a, dtype_b):
        """Addition should return same dtype and values as NumPy."""
        r_a = rp.asarray([1, 2, 3], dtype=dtype_a)
        r_b = rp.asarray([4, 5, 6], dtype=dtype_b)
        n_a = np.array([1, 2, 3], dtype=dtype_a)
        n_b = np.array([4, 5, 6], dtype=dtype_b)

        r_result = r_a + r_b
        n_result = n_a + n_b

        assert r_result.dtype == str(n_result.dtype), f"{dtype_a}+{dtype_b}: Expected {n_result.dtype}, got {r_result.dtype}"
        assert_eq(r_result, n_result)

    @pytest.mark.parametrize("dtype_a", ALL_DTYPES)
    @pytest.mark.parametrize("dtype_b", ALL_DTYPES)
    def test_pow_dtype_and_values(self, dtype_a, dtype_b):
        """Power should return same dtype and values as NumPy."""
        r_a = rp.asarray([2, 3, 4], dtype=dtype_a)
        r_b = rp.asarray([2, 2, 2], dtype=dtype_b)
        n_a = np.array([2, 3, 4], dtype=dtype_a)
        n_b = np.array([2, 2, 2], dtype=dtype_b)

        r_result = r_a ** r_b
        n_result = n_a ** n_b

        assert r_result.dtype == str(n_result.dtype), f"{dtype_a}**{dtype_b}: Expected {n_result.dtype}, got {r_result.dtype}"
        assert_eq(r_result, n_result)


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


class TestLog10Log2:
    """Test log10 and log2 functions."""

    def test_log10_float64(self):
        n = np.array([1.0, 10.0, 100.0, 1000.0])
        r = rp.asarray(n)
        assert_eq(rp.log10(r), np.log10(n))

    def test_log2_float64(self):
        n = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
        r = rp.asarray(n)
        assert_eq(rp.log2(r), np.log2(n))

    def test_log10_float32(self):
        n = np.array([1.0, 10.0, 100.0], dtype="float32")
        r = rp.asarray(n)
        result = rp.log10(r)
        assert result.dtype == "float32"
        assert_eq(result, np.log10(n))

    def test_log2_float32(self):
        n = np.array([1.0, 2.0, 4.0], dtype="float32")
        r = rp.asarray(n)
        result = rp.log2(r)
        assert result.dtype == "float32"
        assert_eq(result, np.log2(n))

    def test_log10_int_promotes(self):
        """log10 of int should promote to float64."""
        n = np.array([1, 10, 100], dtype="int64")
        r = rp.asarray(n)
        r_result = rp.log10(r)
        n_result = np.log10(n)
        assert r_result.dtype == "float64"
        assert_eq(r_result, n_result)


class TestHyperbolic:
    """Test hyperbolic functions sinh, cosh, tanh."""

    def test_sinh_float64(self):
        n = np.array([0.0, 1.0, -1.0, 2.0])
        r = rp.asarray(n)
        assert_eq(rp.sinh(r), np.sinh(n))

    def test_cosh_float64(self):
        n = np.array([0.0, 1.0, -1.0, 2.0])
        r = rp.asarray(n)
        assert_eq(rp.cosh(r), np.cosh(n))

    def test_tanh_float64(self):
        n = np.array([0.0, 1.0, -1.0, 2.0])
        r = rp.asarray(n)
        assert_eq(rp.tanh(r), np.tanh(n))

    def test_sinh_float32(self):
        n = np.array([0.0, 1.0, -1.0], dtype="float32")
        r = rp.asarray(n)
        result = rp.sinh(r)
        assert result.dtype == "float32"
        assert_eq(result, np.sinh(n))

    def test_cosh_float32(self):
        n = np.array([0.0, 1.0, -1.0], dtype="float32")
        r = rp.asarray(n)
        result = rp.cosh(r)
        assert result.dtype == "float32"
        assert_eq(result, np.cosh(n))

    def test_tanh_float32(self):
        n = np.array([0.0, 1.0, -1.0], dtype="float32")
        r = rp.asarray(n)
        result = rp.tanh(r)
        assert result.dtype == "float32"
        assert_eq(result, np.tanh(n))

    def test_hyperbolic_complex(self):
        n = np.array([1+2j, 0+1j, 1+0j])
        r = rp.asarray(n)
        assert_eq(rp.sinh(r), np.sinh(n))
        assert_eq(rp.cosh(r), np.cosh(n))
        assert_eq(rp.tanh(r), np.tanh(n))


class TestSign:
    """Test sign function."""

    def test_sign_float64(self):
        n = np.array([3.0, 0.0, -3.0, 5.5, -2.5])
        r = rp.asarray(n)
        assert_eq(rp.sign(r), np.sign(n))

    def test_sign_float32(self):
        n = np.array([3.0, 0.0, -3.0], dtype="float32")
        r = rp.asarray(n)
        result = rp.sign(r)
        assert result.dtype == "float32"
        assert_eq(result, np.sign(n))

    def test_sign_int64(self):
        n = np.array([3, 0, -3], dtype="int64")
        r = rp.asarray(n)
        result = rp.sign(r)
        assert result.dtype == "int64"
        assert_eq(result, np.sign(n))


class TestIsnanIsinfIsfinite:
    """Test isnan, isinf, isfinite functions."""

    def test_isnan_float64(self):
        n = np.array([1.0, np.nan, 2.0, np.nan])
        r = rp.asarray(n)
        n_result = np.isnan(n)
        r_result = rp.isnan(r)
        # rumpy returns float array with 0/1, numpy returns bool
        assert_eq(r_result, n_result.astype(float))

    def test_isinf_float64(self):
        n = np.array([1.0, np.inf, -np.inf, 0.0])
        r = rp.asarray(n)
        n_result = np.isinf(n)
        r_result = rp.isinf(r)
        assert_eq(r_result, n_result.astype(float))

    def test_isfinite_float64(self):
        n = np.array([1.0, np.inf, np.nan, -np.inf, 0.0])
        r = rp.asarray(n)
        n_result = np.isfinite(n)
        r_result = rp.isfinite(r)
        assert_eq(r_result, n_result.astype(float))

    def test_isnan_int(self):
        """Integers are never NaN."""
        n = np.array([1, 2, 3], dtype="int64")
        r = rp.asarray(n)
        r_result = rp.isnan(r)
        assert_eq(r_result, np.zeros(3))

    def test_isinf_int(self):
        """Integers are never infinite."""
        n = np.array([1, 2, 3], dtype="int64")
        r = rp.asarray(n)
        r_result = rp.isinf(r)
        assert_eq(r_result, np.zeros(3))

    def test_isfinite_int(self):
        """Integers are always finite."""
        n = np.array([1, 2, 3], dtype="int64")
        r = rp.asarray(n)
        r_result = rp.isfinite(r)
        assert_eq(r_result, np.ones(3))


class TestMaximumMinimum:
    """Test maximum and minimum element-wise functions."""

    def test_maximum_float64(self):
        a = np.array([1.0, 5.0, 3.0])
        b = np.array([4.0, 2.0, 3.0])
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.maximum(ra, rb), np.maximum(a, b))

    def test_minimum_float64(self):
        a = np.array([1.0, 5.0, 3.0])
        b = np.array([4.0, 2.0, 3.0])
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.minimum(ra, rb), np.minimum(a, b))

    def test_maximum_int64(self):
        a = np.array([1, 5, 3], dtype="int64")
        b = np.array([4, 2, 3], dtype="int64")
        ra, rb = rp.asarray(a), rp.asarray(b)
        result = rp.maximum(ra, rb)
        assert result.dtype == "int64"
        assert_eq(result, np.maximum(a, b))

    def test_minimum_int64(self):
        a = np.array([1, 5, 3], dtype="int64")
        b = np.array([4, 2, 3], dtype="int64")
        ra, rb = rp.asarray(a), rp.asarray(b)
        result = rp.minimum(ra, rb)
        assert result.dtype == "int64"
        assert_eq(result, np.minimum(a, b))

    def test_maximum_broadcast(self):
        a = np.array([[1.0, 2.0, 3.0]])
        b = np.array([[10.0], [1.0]])
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.maximum(ra, rb), np.maximum(a, b))

    def test_minimum_broadcast(self):
        a = np.array([[1.0, 2.0, 3.0]])
        b = np.array([[10.0], [1.0]])
        ra, rb = rp.asarray(a), rp.asarray(b)
        assert_eq(rp.minimum(ra, rb), np.minimum(a, b))

    def test_maximum_with_nan(self):
        """maximum propagates NaN like NumPy."""
        a = np.array([1.0, np.nan, 3.0])
        b = np.array([2.0, 2.0, np.nan])
        ra, rb = rp.asarray(a), rp.asarray(b)
        r_result = rp.maximum(ra, rb)
        n_result = np.maximum(a, b)
        assert_eq(r_result, n_result)

    def test_minimum_with_nan(self):
        """minimum propagates NaN like NumPy."""
        a = np.array([1.0, np.nan, 3.0])
        b = np.array([2.0, 2.0, np.nan])
        ra, rb = rp.asarray(a), rp.asarray(b)
        r_result = rp.minimum(ra, rb)
        n_result = np.minimum(a, b)
        assert_eq(r_result, n_result)


class TestCountNonzero:
    """Test count_nonzero function."""

    def test_count_nonzero_1d(self):
        n = np.array([0, 1, 0, 2, 3, 0])
        r = rp.asarray(n)
        assert rp.count_nonzero(r) == np.count_nonzero(n)

    def test_count_nonzero_2d(self):
        n = np.array([[0, 1, 2], [3, 0, 0]])
        r = rp.asarray(n)
        assert rp.count_nonzero(r) == np.count_nonzero(n)

    def test_count_nonzero_float(self):
        n = np.array([0.0, 1.5, 0.0, -2.5])
        r = rp.asarray(n)
        assert rp.count_nonzero(r) == np.count_nonzero(n)

    def test_count_nonzero_all_zero(self):
        n = np.zeros(5)
        r = rp.asarray(n)
        assert rp.count_nonzero(r) == 0

    def test_count_nonzero_all_nonzero(self):
        n = np.array([1, 2, 3])
        r = rp.asarray(n)
        assert rp.count_nonzero(r) == 3


class TestComplexAccessors:
    """Test real, imag, conj for complex arrays."""

    def test_real_complex128(self):
        n = np.array([1+2j, 3+4j, 5+6j])
        r = rp.asarray(n)
        assert_eq(rp.real(r), np.real(n))

    def test_imag_complex128(self):
        n = np.array([1+2j, 3+4j, 5+6j])
        r = rp.asarray(n)
        assert_eq(rp.imag(r), np.imag(n))

    def test_conj_complex128(self):
        n = np.array([1+2j, 3+4j, 5+6j])
        r = rp.asarray(n)
        assert_eq(rp.conj(r), np.conj(n))

    def test_real_complex64(self):
        n = np.array([1+2j, 3+4j], dtype="complex64")
        r = rp.asarray(n)
        result = rp.real(r)
        assert result.dtype == "float32"
        assert_eq(result, np.real(n))

    def test_imag_complex64(self):
        n = np.array([1+2j, 3+4j], dtype="complex64")
        r = rp.asarray(n)
        result = rp.imag(r)
        assert result.dtype == "float32"
        assert_eq(result, np.imag(n))

    def test_conj_complex64(self):
        n = np.array([1+2j, 3+4j], dtype="complex64")
        r = rp.asarray(n)
        result = rp.conj(r)
        assert result.dtype == "complex64"
        assert_eq(result, np.conj(n))

    def test_real_float_passthrough(self):
        """real of float array returns the array unchanged."""
        n = np.array([1.0, 2.0, 3.0])
        r = rp.asarray(n)
        assert_eq(rp.real(r), np.real(n))

    def test_imag_float_zeros(self):
        """imag of float array returns zeros."""
        n = np.array([1.0, 2.0, 3.0])
        r = rp.asarray(n)
        assert_eq(rp.imag(r), np.imag(n))

    def test_conj_float_passthrough(self):
        """conj of float array returns the array unchanged."""
        n = np.array([1.0, 2.0, 3.0])
        r = rp.asarray(n)
        assert_eq(rp.conj(r), np.conj(n))

    def test_real_method(self):
        """Test .real property on array."""
        n = np.array([1+2j, 3+4j])
        r = rp.asarray(n)
        assert_eq(r.real, n.real)

    def test_imag_method(self):
        """Test .imag property on array."""
        n = np.array([1+2j, 3+4j])
        r = rp.asarray(n)
        assert_eq(r.imag, n.imag)

    def test_conj_method(self):
        """Test .conj() method on array."""
        n = np.array([1+2j, 3+4j])
        r = rp.asarray(n)
        assert_eq(r.conj(), n.conj())


class TestScalarUfuncs:
    """Test that ufuncs accept scalar (Python float) inputs."""

    def test_sqrt_scalar(self):
        """rp.sqrt(4.0) should return 2.0."""
        assert abs(rp.sqrt(4.0) - np.sqrt(4.0)) < 1e-10

    def test_exp_scalar(self):
        """rp.exp(1.0) should return e."""
        assert abs(rp.exp(1.0) - np.exp(1.0)) < 1e-10

    def test_log_scalar(self):
        """rp.log(e) should return 1.0."""
        import math
        assert abs(rp.log(math.e) - np.log(math.e)) < 1e-10

    def test_sin_scalar(self):
        assert abs(rp.sin(0.5) - np.sin(0.5)) < 1e-10

    def test_cos_scalar(self):
        assert abs(rp.cos(0.5) - np.cos(0.5)) < 1e-10

    def test_tan_scalar(self):
        assert abs(rp.tan(0.5) - np.tan(0.5)) < 1e-10

    def test_tanh_scalar(self):
        assert abs(rp.tanh(0.5) - np.tanh(0.5)) < 1e-10

    def test_abs_scalar(self):
        assert abs(rp.abs(-3.14) - np.abs(-3.14)) < 1e-10

    def test_floor_scalar(self):
        assert abs(rp.floor(3.7) - np.floor(3.7)) < 1e-10

    def test_ceil_scalar(self):
        assert abs(rp.ceil(3.2) - np.ceil(3.2)) < 1e-10

    def test_log10_scalar(self):
        assert abs(rp.log10(100.0) - np.log10(100.0)) < 1e-10

    def test_log2_scalar(self):
        assert abs(rp.log2(8.0) - np.log2(8.0)) < 1e-10

    def test_arcsin_scalar(self):
        assert abs(rp.arcsin(0.5) - np.arcsin(0.5)) < 1e-10

    def test_arccos_scalar(self):
        assert abs(rp.arccos(0.5) - np.arccos(0.5)) < 1e-10

    def test_arctan_scalar(self):
        assert abs(rp.arctan(1.0) - np.arctan(1.0)) < 1e-10

    def test_sign_scalar(self):
        assert rp.sign(-5.0) == np.sign(-5.0)
        assert rp.sign(5.0) == np.sign(5.0)
        assert rp.sign(0.0) == np.sign(0.0)


class TestArrayUfunc:
    """Test __array_ufunc__ - numpy should defer to rumpy."""

    def test_numpy_float64_times_rumpy_array(self):
        """np.float64 * rp.array should return rumpy array."""
        x = rp.array([1.0, 2.0, 3.0])
        scalar = np.float64(2.0)
        result = scalar * x
        assert isinstance(result, rp.ndarray), f"Expected rp.ndarray, got {type(result)}"
        assert_eq(result, np.array([2.0, 4.0, 6.0]))

    def test_numpy_sqrt_times_rumpy_array(self):
        """np.sqrt(2) * rp.array should return rumpy array."""
        x = rp.array([1.0, 2.0, 3.0])
        scalar = np.sqrt(2.0)  # Returns numpy.float64
        result = scalar * x
        assert isinstance(result, rp.ndarray), f"Expected rp.ndarray, got {type(result)}"
        expected = np.sqrt(2.0) * np.array([1.0, 2.0, 3.0])
        assert_eq(result, expected)

    def test_gelu_pattern(self):
        """GELU-like pattern with numpy.float64 * rumpy array."""
        x = rp.array([0.0, 1.0, 2.0])
        sqrt_2_pi = np.sqrt(2.0 / np.pi)  # numpy.float64
        result = sqrt_2_pi * x
        assert isinstance(result, rp.ndarray), f"Expected rp.ndarray, got {type(result)}"
        expected = sqrt_2_pi * np.array([0.0, 1.0, 2.0])
        assert_eq(result, expected)

    def test_numpy_add_defers(self):
        """np.add(np.float64, rp.array) should return rumpy array."""
        x = rp.array([1.0, 2.0, 3.0])
        result = np.add(np.float64(10.0), x)
        assert isinstance(result, rp.ndarray), f"Expected rp.ndarray, got {type(result)}"
        assert_eq(result, np.array([11.0, 12.0, 13.0]))
