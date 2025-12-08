"""Tests for numerical operations - gradient, trapezoid, interp, correlate.

Parametrizes over dtypes and shapes for comprehensive coverage.
See designs/testing.md for testing philosophy.
"""

import numpy as np
import pytest

import rumpy as rp
from conftest import CORE_SHAPES, FLOAT_DTYPES
from helpers import assert_eq, make_pair


# === Gradient Tests ===


class TestGradient:
    """Test numerical gradient computation.

    NOTE: Currently only float64 is tested due to a misaligned pointer bug in the
    Rust implementation when using float32 (see src/ops/numerical.rs line 22).
    """

    def test_1d_uniform_spacing(self):
        """Test 1D gradient with uniform spacing."""
        n = np.array([1, 2, 4, 7, 11], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.gradient(r), np.gradient(n))

    def test_1d_with_scalar_spacing(self):
        """Test 1D gradient with explicit scalar spacing."""
        n = np.array([1, 2, 4, 7, 11], dtype=np.float64)
        r = rp.asarray(n)
        spacing = 2.0
        assert_eq(rp.gradient(r, spacing), np.gradient(n, spacing))

    def test_1d_with_coord_array(self):
        """Test 1D gradient with non-uniform coordinate array."""
        f_data = [1, 2, 4, 7, 11]
        x_data = [0, 1, 3, 6, 10]  # Non-uniform spacing
        n_f = np.array(f_data, dtype=np.float64)
        n_x = np.array(x_data, dtype=np.float64)
        r_f = rp.asarray(n_f)
        r_x = rp.asarray(n_x)
        assert_eq(rp.gradient(r_f, r_x), np.gradient(n_f, n_x))

    @pytest.mark.parametrize("shape", [(5,), (8,), (12,)])
    def test_1d_shapes(self, shape):
        """Test 1D gradient with various array sizes."""
        r, n = make_pair(shape, "float64")
        assert_eq(rp.gradient(r), np.gradient(n))

    def test_2d_default(self):
        """Test 2D gradient returns gradients along each axis."""
        n = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        r = rp.asarray(n)
        r_grads = rp.gradient(r)
        n_grads = np.gradient(n)
        # Returns tuple of gradients
        assert len(r_grads) == len(n_grads) == 2
        assert_eq(r_grads[0], n_grads[0])
        assert_eq(r_grads[1], n_grads[1])

    @pytest.mark.parametrize("axis", [0, 1])
    def test_2d_along_axis(self, axis):
        """Test 2D gradient along specific axis."""
        n = np.array([[1, 2, 4], [4, 6, 8]], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(rp.gradient(r, axis=axis), np.gradient(n, axis=axis))

    @pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4)])
    def test_multidim_shapes(self, shape):
        """Test gradient with multidimensional shapes (returns list)."""
        r, n = make_pair(shape, "float64")
        r_grads = rp.gradient(r)
        n_grads = np.gradient(n)
        assert len(r_grads) == len(n_grads)
        for r_g, n_g in zip(r_grads, n_grads):
            assert_eq(r_g, n_g)

    def test_two_elements(self):
        """Gradient of two elements uses forward/backward difference."""
        n = np.array([1.0, 3.0])
        r = rp.asarray(n)
        assert_eq(rp.gradient(r), np.gradient(n))

    def test_constant_array(self):
        """Gradient of constant array should be zero."""
        n = np.array([5.0, 5.0, 5.0, 5.0])
        r = rp.asarray(n)
        expected = np.zeros_like(n)
        assert_eq(rp.gradient(r), expected)

    def test_linear_function(self):
        """Gradient of linear function should be constant."""
        n = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        r = rp.asarray(n)
        result = rp.gradient(r)
        # Gradient of y=x with dx=1 should be 1
        assert_eq(result, np.ones_like(n))

    def test_single_element_error(self):
        """Gradient of single element should raise error like numpy."""
        n = np.array([5.0])
        r = rp.asarray(n)
        with pytest.raises(ValueError):
            rp.gradient(r)


# === Trapezoid Integration Tests ===


class TestTrapezoid:
    """Test trapezoidal integration."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_simple_integration(self, dtype):
        """Simple integration of y values with default spacing."""
        n = np.array([1, 2, 3, 4], dtype=dtype)
        r = rp.asarray(n)
        r_result = float(rp.trapezoid(r))
        n_result = float(np.trapezoid(n))
        assert abs(r_result - n_result) < 1e-10

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_with_x_coordinates(self, dtype):
        """Integration with explicit x coordinates (non-uniform)."""
        y_data = [1, 2, 3]
        x_data = [0, 1, 3]  # Non-uniform spacing
        n_y = np.array(y_data, dtype=dtype)
        n_x = np.array(x_data, dtype=dtype)
        r_y = rp.asarray(n_y)
        r_x = rp.asarray(n_x)
        r_result = float(rp.trapezoid(r_y, r_x))
        n_result = float(np.trapezoid(n_y, n_x))
        assert abs(r_result - n_result) < 1e-10

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("dx", [0.5, 1.0, 2.0])
    def test_with_dx_spacing(self, dtype, dx):
        """Integration with explicit uniform spacing."""
        n = np.array([1, 2, 3, 4], dtype=dtype)
        r = rp.asarray(n)
        r_result = float(rp.trapezoid(r, dx=dx))
        n_result = float(np.trapezoid(n, dx=dx))
        assert abs(r_result - n_result) < 1e-10

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_2d_default_axis(self, dtype):
        """2D integration along default axis (-1)."""
        n = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(rp.trapezoid(r), np.trapezoid(n))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_2d_along_axis(self, dtype, axis):
        """2D integration along specific axis."""
        n = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
        r = rp.asarray(n)
        assert_eq(rp.trapezoid(r, axis=axis), np.trapezoid(n, axis=axis))

    @pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4)])
    def test_multidim_shapes(self, shape):
        """Test trapezoid integration with multidimensional shapes."""
        r, n = make_pair(shape, "float64")
        assert_eq(rp.trapezoid(r), np.trapezoid(n))

    def test_empty_array(self):
        """Empty array gives 0."""
        n = np.array([])
        r = rp.asarray(n)
        assert float(rp.trapezoid(r)) == 0.0

    def test_single_element(self):
        """Single element gives 0."""
        n = np.array([5.0])
        r = rp.asarray(n)
        assert float(rp.trapezoid(r)) == 0.0

    def test_constant_function(self):
        """Integral of constant function."""
        # y = 3, x = [0, 1, 2, 3, 4]
        # Integral should be 3 * 4 = 12
        n = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        r = rp.asarray(n)
        result = float(rp.trapezoid(r))
        expected = 12.0
        assert abs(result - expected) < 1e-10

    def test_linear_function(self):
        """Integral of linear function y = x."""
        # y = x, x = [0, 1, 2, 3, 4]
        # Integral from 0 to 4 should be 0.5 * 4^2 = 8
        n = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        r = rp.asarray(n)
        result = float(rp.trapezoid(r))
        expected = 8.0
        assert abs(result - expected) < 1e-10


# === Interpolation Tests ===


class TestInterp:
    """Test 1D linear interpolation."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic_interpolation(self, dtype):
        """Basic linear interpolation."""
        xp_data = [0, 1, 2]
        fp_data = [0, 1, 4]
        x_data = [0.5, 1.5]
        n_xp = np.array(xp_data, dtype=dtype)
        n_fp = np.array(fp_data, dtype=dtype)
        n_x = np.array(x_data, dtype=dtype)
        r_xp = rp.asarray(n_xp)
        r_fp = rp.asarray(n_fp)
        r_x = rp.asarray(n_x)
        assert_eq(rp.interp(r_x, r_xp, r_fp), np.interp(n_x, n_xp, n_fp))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_extrapolation_left(self, dtype):
        """Extrapolation to left returns first value by default."""
        xp_data = [1, 2, 3]
        fp_data = [10, 20, 30]
        x_data = [0]
        n_xp = np.array(xp_data, dtype=dtype)
        n_fp = np.array(fp_data, dtype=dtype)
        n_x = np.array(x_data, dtype=dtype)
        r_xp = rp.asarray(n_xp)
        r_fp = rp.asarray(n_fp)
        r_x = rp.asarray(n_x)
        assert_eq(rp.interp(r_x, r_xp, r_fp), np.interp(n_x, n_xp, n_fp))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_extrapolation_right(self, dtype):
        """Extrapolation to right returns last value by default."""
        xp_data = [1, 2, 3]
        fp_data = [10, 20, 30]
        x_data = [5]
        n_xp = np.array(xp_data, dtype=dtype)
        n_fp = np.array(fp_data, dtype=dtype)
        n_x = np.array(x_data, dtype=dtype)
        r_xp = rp.asarray(n_xp)
        r_fp = rp.asarray(n_fp)
        r_x = rp.asarray(n_x)
        assert_eq(rp.interp(r_x, r_xp, r_fp), np.interp(n_x, n_xp, n_fp))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_exact_match(self, dtype):
        """Interpolation at exact data points returns exact values."""
        xp_data = [0, 1, 2]
        fp_data = [0, 10, 20]
        x_data = [0, 1, 2]
        n_xp = np.array(xp_data, dtype=dtype)
        n_fp = np.array(fp_data, dtype=dtype)
        n_x = np.array(x_data, dtype=dtype)
        r_xp = rp.asarray(n_xp)
        r_fp = rp.asarray(n_fp)
        r_x = rp.asarray(n_x)
        assert_eq(rp.interp(r_x, r_xp, r_fp), np.interp(n_x, n_xp, n_fp))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_with_left_right(self, dtype):
        """Custom left/right extrapolation values."""
        xp_data = [1, 2, 3]
        fp_data = [10, 20, 30]
        x_data = [0, 4]
        n_xp = np.array(xp_data, dtype=dtype)
        n_fp = np.array(fp_data, dtype=dtype)
        n_x = np.array(x_data, dtype=dtype)
        r_xp = rp.asarray(n_xp)
        r_fp = rp.asarray(n_fp)
        r_x = rp.asarray(n_x)
        left, right = -1.0, -2.0
        assert_eq(
            rp.interp(r_x, r_xp, r_fp, left=left, right=right),
            np.interp(n_x, n_xp, n_fp, left=left, right=right)
        )

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_scalar_x(self, dtype):
        """Single x value interpolation."""
        xp_data = [0, 1, 2]
        fp_data = [0, 10, 20]
        x_data = [0.5]
        n_xp = np.array(xp_data, dtype=dtype)
        n_fp = np.array(fp_data, dtype=dtype)
        n_x = np.array(x_data, dtype=dtype)
        r_xp = rp.asarray(n_xp)
        r_fp = rp.asarray(n_fp)
        r_x = rp.asarray(n_x)
        assert_eq(rp.interp(r_x, r_xp, r_fp), np.interp(n_x, n_xp, n_fp))

    def test_constant_function(self):
        """Interpolation of constant function."""
        n_xp = np.array([0.0, 1.0, 2.0, 3.0])
        n_fp = np.array([5.0, 5.0, 5.0, 5.0])
        n_x = np.array([0.5, 1.5, 2.5])
        r_xp = rp.asarray(n_xp)
        r_fp = rp.asarray(n_fp)
        r_x = rp.asarray(n_x)
        result = rp.interp(r_x, r_xp, r_fp)
        expected = np.array([5.0, 5.0, 5.0])
        assert_eq(result, expected)

    def test_linear_function(self):
        """Interpolation of linear function y = 2x."""
        n_xp = np.array([0.0, 1.0, 2.0, 3.0])
        n_fp = np.array([0.0, 2.0, 4.0, 6.0])
        n_x = np.array([0.5, 1.5, 2.5])
        r_xp = rp.asarray(n_xp)
        r_fp = rp.asarray(n_fp)
        r_x = rp.asarray(n_x)
        result = rp.interp(r_x, r_xp, r_fp)
        expected = np.array([1.0, 3.0, 5.0])
        assert_eq(result, expected)

    def test_non_uniform_spacing(self):
        """Interpolation with non-uniform x spacing."""
        n_xp = np.array([0.0, 1.0, 3.0, 6.0])
        n_fp = np.array([0.0, 1.0, 3.0, 6.0])
        n_x = np.array([0.5, 2.0, 4.5])
        r_xp = rp.asarray(n_xp)
        r_fp = rp.asarray(n_fp)
        r_x = rp.asarray(n_x)
        assert_eq(rp.interp(r_x, r_xp, r_fp), np.interp(n_x, n_xp, n_fp))


# === Correlation Tests ===


class TestCorrelate:
    """Test cross-correlation."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("mode", ["full", "same", "valid"])
    def test_modes(self, dtype, mode):
        """Test correlation with different modes."""
        a_data = [1, 2, 3, 4, 5]
        v_data = [0.25, 0.5, 0.25]
        n_a = np.array(a_data, dtype=dtype)
        n_v = np.array(v_data, dtype=dtype)
        r_a = rp.asarray(n_a)
        r_v = rp.asarray(n_v)
        assert_eq(rp.correlate(r_a, r_v, mode), np.correlate(n_a, n_v, mode))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_full_mode(self, dtype):
        """Full cross-correlation."""
        a_data = [1, 2, 3]
        v_data = [0.5, 0.5]
        n_a = np.array(a_data, dtype=dtype)
        n_v = np.array(v_data, dtype=dtype)
        r_a = rp.asarray(n_a)
        r_v = rp.asarray(n_v)
        assert_eq(rp.correlate(r_a, r_v, "full"), np.correlate(n_a, n_v, "full"))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_same_mode(self, dtype):
        """Same-size cross-correlation."""
        a_data = [1, 2, 3, 4, 5]
        v_data = [0.25, 0.5, 0.25]
        n_a = np.array(a_data, dtype=dtype)
        n_v = np.array(v_data, dtype=dtype)
        r_a = rp.asarray(n_a)
        r_v = rp.asarray(n_v)
        assert_eq(rp.correlate(r_a, r_v, "same"), np.correlate(n_a, n_v, "same"))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_valid_mode(self, dtype):
        """Valid cross-correlation (no padding)."""
        a_data = [1, 2, 3, 4, 5]
        v_data = [1, 2, 1]
        n_a = np.array(a_data, dtype=dtype)
        n_v = np.array(v_data, dtype=dtype)
        r_a = rp.asarray(n_a)
        r_v = rp.asarray(n_v)
        assert_eq(rp.correlate(r_a, r_v, "valid"), np.correlate(n_a, n_v, "valid"))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_autocorrelation(self, dtype):
        """Autocorrelation (array with itself)."""
        a_data = [1, 2, 3]
        n_a = np.array(a_data, dtype=dtype)
        r_a = rp.asarray(n_a)
        assert_eq(rp.correlate(r_a, r_a, "full"), np.correlate(n_a, n_a, "full"))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_different_from_convolve(self, dtype):
        """Correlate differs from convolve (v is not reversed)."""
        a_data = [1, 2, 3]
        v_data = [1, 2]  # Asymmetric
        n_a = np.array(a_data, dtype=dtype)
        n_v = np.array(v_data, dtype=dtype)
        r_a = rp.asarray(n_a)
        r_v = rp.asarray(n_v)

        r_corr = rp.correlate(r_a, r_v, "full")
        r_conv = rp.convolve(r_a, r_v, "full")
        n_corr = np.correlate(n_a, n_v, "full")
        n_conv = np.convolve(n_a, n_v, "full")

        assert_eq(r_corr, n_corr)
        assert_eq(r_conv, n_conv)
        # They should be different when v is asymmetric
        assert not np.allclose(np.asarray(r_corr), np.asarray(r_conv))

    def test_constant_signal(self):
        """Correlation of constant signals."""
        n_a = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        n_v = np.array([1.0, 1.0, 1.0])
        r_a = rp.asarray(n_a)
        r_v = rp.asarray(n_v)
        assert_eq(rp.correlate(r_a, r_v, "same"), np.correlate(n_a, n_v, "same"))

    def test_single_element(self):
        """Correlation with single element."""
        n_a = np.array([1.0, 2.0, 3.0])
        n_v = np.array([2.0])
        r_a = rp.asarray(n_a)
        r_v = rp.asarray(n_v)
        assert_eq(rp.correlate(r_a, r_v, "same"), np.correlate(n_a, n_v, "same"))


class TestConvolve:
    """Test convolve function."""

    def test_full_mode(self):
        a = rp.array([1.0, 2.0, 3.0])
        v = rp.array([0.5, 0.5])
        r = rp.convolve(a, v, 'full')
        n = np.convolve([1, 2, 3], [0.5, 0.5], 'full')
        assert_eq(r, n)

    def test_same_mode(self):
        a = rp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        v = rp.array([0.25, 0.5, 0.25])
        r = rp.convolve(a, v, 'same')
        n = np.convolve([1, 2, 3, 4, 5], [0.25, 0.5, 0.25], 'same')
        assert_eq(r, n)

    def test_valid_mode(self):
        a = rp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        v = rp.array([1.0, 2.0, 1.0])
        r = rp.convolve(a, v, 'valid')
        n = np.convolve([1, 2, 3, 4, 5], [1, 2, 1], 'valid')
        assert_eq(r, n)
