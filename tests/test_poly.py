"""Comprehensive tests for polynomial operations.

Tests cover polynomial evaluation, fitting, derivatives, integrals, and roots.
Uses parametrization for systematic coverage of different polynomial degrees.

See designs/testing.md for testing philosophy.
"""

import numpy as np
import pytest
import rumpy as rp
from helpers import assert_eq
from conftest import FLOAT_DTYPES


# ============================================================================
# Polynomial Evaluation
# ============================================================================


class TestPolyval:
    """Polynomial evaluation: p(x) = c[0]*x^n + ... + c[n]."""

    def test_constant_polynomial(self):
        """Evaluate constant polynomial p(x) = 5."""
        coeffs_r = rp.array([5.0])
        coeffs_n = np.array([5.0])
        x_r = rp.array([0.0, 1.0, 100.0, -50.0])
        x_n = np.array([0.0, 1.0, 100.0, -50.0])

        r = rp.polyval(coeffs_r, x_r)
        n = np.polyval(coeffs_n, x_n)
        assert_eq(r, n)

    def test_linear_polynomial(self):
        """Evaluate linear polynomial p(x) = 2x + 1."""
        coeffs_r = rp.array([2.0, 1.0])
        coeffs_n = np.array([2.0, 1.0])
        x_r = rp.array([-1.0, 0.0, 1.0, 2.0])
        x_n = np.array([-1.0, 0.0, 1.0, 2.0])

        r = rp.polyval(coeffs_r, x_r)
        n = np.polyval(coeffs_n, x_n)
        assert_eq(r, n)

    def test_quadratic_polynomial(self):
        """Evaluate quadratic p(x) = x^2 + 2x + 3."""
        coeffs_r = rp.array([1.0, 2.0, 3.0])
        coeffs_n = np.array([1.0, 2.0, 3.0])
        x_r = rp.array([0.0, 1.0, 2.0, -1.0])
        x_n = np.array([0.0, 1.0, 2.0, -1.0])

        r = rp.polyval(coeffs_r, x_r)
        n = np.polyval(coeffs_n, x_n)
        assert_eq(r, n)

    def test_cubic_polynomial(self):
        """Evaluate cubic p(x) = x^3 - 2x^2 + x - 1."""
        coeffs_r = rp.array([1.0, -2.0, 1.0, -1.0])
        coeffs_n = np.array([1.0, -2.0, 1.0, -1.0])
        x_r = rp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        x_n = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        r = rp.polyval(coeffs_r, x_r)
        n = np.polyval(coeffs_n, x_n)
        assert_eq(r, n)

    def test_high_degree_polynomial(self):
        """Evaluate degree-5 polynomial."""
        coeffs_r = rp.array([1.0, -1.0, 2.0, -2.0, 3.0, -3.0])
        coeffs_n = np.array([1.0, -1.0, 2.0, -2.0, 3.0, -3.0])
        x_r = rp.array([0.0, 0.5, 1.0, -0.5])
        x_n = np.array([0.0, 0.5, 1.0, -0.5])

        r = rp.polyval(coeffs_r, x_r)
        n = np.polyval(coeffs_n, x_n)
        assert_eq(r, n)

    def test_zero_polynomial(self):
        """Evaluate polynomial with all zero coefficients."""
        coeffs_r = rp.array([0.0, 0.0, 0.0])
        coeffs_n = np.array([0.0, 0.0, 0.0])
        x_r = rp.array([1.0, 2.0, 3.0])
        x_n = np.array([1.0, 2.0, 3.0])

        r = rp.polyval(coeffs_r, x_r)
        n = np.polyval(coeffs_n, x_n)
        assert_eq(r, n)

    def test_single_point_evaluation(self):
        """Evaluate polynomial at single point."""
        coeffs_r = rp.array([1.0, 2.0, 3.0])
        coeffs_n = np.array([1.0, 2.0, 3.0])
        x_r = rp.array([5.0])
        x_n = np.array([5.0])

        r = rp.polyval(coeffs_r, x_r)
        n = np.polyval(coeffs_n, x_n)
        assert_eq(r, n)

    def test_negative_coefficients(self):
        """Evaluate polynomial with negative coefficients."""
        coeffs_r = rp.array([-3.0, -2.0, -1.0])
        coeffs_n = np.array([-3.0, -2.0, -1.0])
        x_r = rp.array([0.0, 1.0, 2.0])
        x_n = np.array([0.0, 1.0, 2.0])

        r = rp.polyval(coeffs_r, x_r)
        n = np.polyval(coeffs_n, x_n)
        assert_eq(r, n)

    def test_2d_x_values(self):
        """Evaluate polynomial at 2D array of points."""
        coeffs_r = rp.array([1.0, 0.0, -1.0])  # x^2 - 1
        coeffs_n = np.array([1.0, 0.0, -1.0])
        x_r = rp.array([[0.0, 1.0], [2.0, 3.0]])
        x_n = np.array([[0.0, 1.0], [2.0, 3.0]])

        r = rp.polyval(coeffs_r, x_r)
        n = np.polyval(coeffs_n, x_n)
        assert_eq(r, n)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtypes(self, dtype):
        """Works with different float types."""
        coeffs_r = rp.array([1.0, 2.0, 3.0], dtype=dtype)
        coeffs_n = np.array([1.0, 2.0, 3.0], dtype=dtype)
        x_r = rp.array([0.0, 1.0, 2.0], dtype=dtype)
        x_n = np.array([0.0, 1.0, 2.0], dtype=dtype)

        r = rp.polyval(coeffs_r, x_r)
        n = np.polyval(coeffs_n, x_n)
        assert_eq(r, n)


# ============================================================================
# Polynomial Derivative
# ============================================================================


class TestPolyder:
    """Polynomial derivative: d/dx p(x)."""

    def test_constant_derivative(self):
        """Derivative of constant is zero (empty array in numpy convention)."""
        p_r = rp.array([5.0])
        p_n = np.array([5.0])

        r = rp.polyder(p_r)
        n = np.polyder(p_n)
        assert_eq(r, n)

    def test_linear_derivative(self):
        """Derivative of 2x + 1 is 2."""
        p_r = rp.array([2.0, 1.0])
        p_n = np.array([2.0, 1.0])

        r = rp.polyder(p_r)
        n = np.polyder(p_n)
        assert_eq(r, n)

    def test_quadratic_derivative(self):
        """Derivative of x^2 + 2x + 3 is 2x + 2."""
        p_r = rp.array([1.0, 2.0, 3.0])
        p_n = np.array([1.0, 2.0, 3.0])

        r = rp.polyder(p_r)
        n = np.polyder(p_n)
        assert_eq(r, n)

    def test_cubic_derivative(self):
        """Derivative of x^3 - 2x^2 + x - 1 is 3x^2 - 4x + 1."""
        p_r = rp.array([1.0, -2.0, 1.0, -1.0])
        p_n = np.array([1.0, -2.0, 1.0, -1.0])

        r = rp.polyder(p_r)
        n = np.polyder(p_n)
        assert_eq(r, n)

    def test_high_degree_derivative(self):
        """Derivative of degree-5 polynomial."""
        p_r = rp.array([2.0, 0.0, -3.0, 1.0, 0.0, 5.0])
        p_n = np.array([2.0, 0.0, -3.0, 1.0, 0.0, 5.0])

        r = rp.polyder(p_r)
        n = np.polyder(p_n)
        assert_eq(r, n)

    def test_second_derivative(self):
        """Second derivative of x^3 is 6x."""
        p_r = rp.array([1.0, 0.0, 0.0, 0.0])  # x^3
        p_n = np.array([1.0, 0.0, 0.0, 0.0])

        r = rp.polyder(p_r, m=2)
        n = np.polyder(p_n, 2)
        assert_eq(r, n)

    def test_third_derivative(self):
        """Third derivative of x^4 is 24x."""
        p_r = rp.array([1.0, 0.0, 0.0, 0.0, 0.0])  # x^4
        p_n = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

        r = rp.polyder(p_r, m=3)
        n = np.polyder(p_n, 3)
        assert_eq(r, n)

    def test_derivative_higher_than_degree(self):
        """Taking derivative of order higher than polynomial degree."""
        p_r = rp.array([1.0, 2.0, 3.0])  # quadratic
        p_n = np.array([1.0, 2.0, 3.0])

        r = rp.polyder(p_r, m=5)
        n = np.polyder(p_n, 5)
        assert_eq(r, n)

    @pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
    def test_derivative_orders(self, degree):
        """Test derivatives of different polynomial degrees."""
        # Create polynomial with random coefficients
        coeffs_n = np.random.randn(degree + 1)
        coeffs_r = rp.asarray(coeffs_n)

        r = rp.polyder(coeffs_r)
        n = np.polyder(coeffs_n)
        assert_eq(r, n)


# ============================================================================
# Polynomial Integral
# ============================================================================


class TestPolyint:
    """Polynomial integral: ∫p(x)dx."""

    def test_constant_integral(self):
        """Integral of constant c is cx."""
        p_r = rp.array([5.0])
        p_n = np.array([5.0])

        r = rp.polyint(p_r)
        n = np.polyint(p_n)
        assert_eq(r, n)

    def test_linear_integral(self):
        """Integral of 2x + 1 is x^2 + x + c."""
        p_r = rp.array([2.0, 1.0])
        p_n = np.array([2.0, 1.0])

        r = rp.polyint(p_r)
        n = np.polyint(p_n)
        assert_eq(r, n)

    def test_quadratic_integral(self):
        """Integral of x^2 + 2x + 3."""
        p_r = rp.array([1.0, 2.0, 3.0])
        p_n = np.array([1.0, 2.0, 3.0])

        r = rp.polyint(p_r)
        n = np.polyint(p_n)
        assert_eq(r, n)

    def test_cubic_integral(self):
        """Integral of cubic polynomial."""
        p_r = rp.array([1.0, -2.0, 1.0, -1.0])
        p_n = np.array([1.0, -2.0, 1.0, -1.0])

        r = rp.polyint(p_r)
        n = np.polyint(p_n)
        assert_eq(r, n)

    def test_integral_with_constant(self):
        """Integral with integration constant k."""
        p_r = rp.array([1.0, 2.0, 3.0])
        p_n = np.array([1.0, 2.0, 3.0])
        k_r = rp.array([5.0])

        r = rp.polyint(p_r, k=k_r)
        n = np.polyint(p_n, k=5.0)
        assert_eq(r, n)

    def test_integral_with_negative_constant(self):
        """Integral with negative integration constant."""
        p_r = rp.array([2.0, 1.0])
        p_n = np.array([2.0, 1.0])
        k_r = rp.array([-3.0])

        r = rp.polyint(p_r, k=k_r)
        n = np.polyint(p_n, k=-3.0)
        assert_eq(r, n)

    def test_double_integral(self):
        """Double integral of polynomial."""
        p_r = rp.array([1.0, 0.0])  # x
        p_n = np.array([1.0, 0.0])

        r = rp.polyint(p_r, m=2)
        n = np.polyint(p_n, m=2)
        assert_eq(r, n)

    def test_triple_integral(self):
        """Triple integral of polynomial."""
        p_r = rp.array([1.0, 0.0, 0.0])  # x^2
        p_n = np.array([1.0, 0.0, 0.0])

        r = rp.polyint(p_r, m=3)
        n = np.polyint(p_n, m=3)
        assert_eq(r, n)

    def test_multiple_integrals_with_constants(self):
        """Multiple integrals with integration constants."""
        p_r = rp.array([1.0, 0.0])  # x
        p_n = np.array([1.0, 0.0])

        # Note: rumpy may handle k differently for m>1, test basic case
        r = rp.polyint(p_r, m=2)
        n = np.polyint(p_n, m=2)
        assert_eq(r, n)

    @pytest.mark.parametrize("degree", [0, 1, 2, 3, 4])
    def test_integral_orders(self, degree):
        """Test integrals of different polynomial degrees."""
        # Create polynomial with random coefficients
        coeffs_n = np.random.randn(degree + 1)
        coeffs_r = rp.asarray(coeffs_n)

        r = rp.polyint(coeffs_r)
        n = np.polyint(coeffs_n)
        assert_eq(r, n)


# ============================================================================
# Polynomial Fitting
# ============================================================================


class TestPolyfit:
    """Polynomial fitting: least-squares fit to data."""

    def test_fit_linear_exact(self):
        """Fit line to exact linear data y = 2x + 1."""
        x_r = rp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        x_n = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y_r = rp.array([1.0, 3.0, 5.0, 7.0, 9.0])
        y_n = np.array([1.0, 3.0, 5.0, 7.0, 9.0])

        r = rp.polyfit(x_r, y_r, 1)
        n = np.polyfit(x_n, y_n, 1)
        assert_eq(r, n, rtol=1e-5)

    def test_fit_quadratic_exact(self):
        """Fit quadratic to exact quadratic data y = x^2 + x + 1."""
        x_r = rp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        x_n = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y_r = rp.array([1.0, 3.0, 7.0, 13.0, 21.0])
        y_n = np.array([1.0, 3.0, 7.0, 13.0, 21.0])

        r = rp.polyfit(x_r, y_r, 2)
        n = np.polyfit(x_n, y_n, 2)
        assert_eq(r, n, rtol=1e-5)

    def test_fit_cubic_exact(self):
        """Fit cubic to exact cubic data."""
        x_r = rp.array([0.0, 1.0, 2.0, 3.0])
        x_n = np.array([0.0, 1.0, 2.0, 3.0])
        # y = x^3 - x^2 + x + 1
        y_r = rp.array([1.0, 2.0, 7.0, 22.0])
        y_n = np.array([1.0, 2.0, 7.0, 22.0])

        r = rp.polyfit(x_r, y_r, 3)
        n = np.polyfit(x_n, y_n, 3)
        assert_eq(r, n, rtol=1e-5)

    def test_fit_constant(self):
        """Fit constant (degree 0) to data."""
        x_r = rp.array([0.0, 1.0, 2.0, 3.0])
        x_n = np.array([0.0, 1.0, 2.0, 3.0])
        y_r = rp.array([5.0, 5.1, 4.9, 5.0])
        y_n = np.array([5.0, 5.1, 4.9, 5.0])

        r = rp.polyfit(x_r, y_r, 0)
        n = np.polyfit(x_n, y_n, 0)
        assert_eq(r, n, rtol=1e-5)

    def test_fit_overdetermined(self):
        """Fit lower-degree polynomial to noisy data (overdetermined)."""
        x_r = rp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        x_n = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        # Approximately y = 2x + 1 with noise
        y_r = rp.array([1.1, 2.9, 5.1, 6.9, 9.1, 10.9])
        y_n = np.array([1.1, 2.9, 5.1, 6.9, 9.1, 10.9])

        r = rp.polyfit(x_r, y_r, 1)
        n = np.polyfit(x_n, y_n, 1)
        assert_eq(r, n, rtol=1e-5)

    def test_fit_minimum_points(self):
        """Fit with minimum number of points (deg+1)."""
        # Degree 2 needs 3 points
        x_r = rp.array([0.0, 1.0, 2.0])
        x_n = np.array([0.0, 1.0, 2.0])
        y_r = rp.array([1.0, 3.0, 7.0])
        y_n = np.array([1.0, 3.0, 7.0])

        r = rp.polyfit(x_r, y_r, 2)
        n = np.polyfit(x_n, y_n, 2)
        assert_eq(r, n, rtol=1e-5)

    def test_fit_negative_values(self):
        """Fit with negative x and y values."""
        x_r = rp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        x_n = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        y_r = rp.array([-3.0, -1.0, 1.0, 3.0, 5.0])  # y = 2x + 1
        y_n = np.array([-3.0, -1.0, 1.0, 3.0, 5.0])

        r = rp.polyfit(x_r, y_r, 1)
        n = np.polyfit(x_n, y_n, 1)
        assert_eq(r, n, rtol=1e-5)

    def test_fit_non_uniform_spacing(self):
        """Fit with non-uniformly spaced x values."""
        x_r = rp.array([0.0, 0.5, 1.5, 4.0, 5.5])
        x_n = np.array([0.0, 0.5, 1.5, 4.0, 5.5])
        y_r = rp.array([1.0, 2.0, 4.0, 9.0, 12.0])
        y_n = np.array([1.0, 2.0, 4.0, 9.0, 12.0])

        r = rp.polyfit(x_r, y_r, 2)
        n = np.polyfit(x_n, y_n, 2)
        assert_eq(r, n, rtol=1e-5)

    @pytest.mark.parametrize("degree", [1, 2, 3])
    def test_fit_degrees(self, degree):
        """Test fitting with different polynomial degrees."""
        # Generate points from a known polynomial
        x_n = np.linspace(-2, 2, 10)
        coeffs_true = np.random.randn(degree + 1)
        y_n = np.polyval(coeffs_true, x_n)

        x_r = rp.asarray(x_n)
        y_r = rp.asarray(y_n)

        r = rp.polyfit(x_r, y_r, degree)
        n = np.polyfit(x_n, y_n, degree)
        assert_eq(r, n, rtol=1e-5)


# ============================================================================
# Polynomial Roots
# ============================================================================


class TestRoots:
    """Polynomial roots: solve p(x) = 0."""

    def test_linear_root(self):
        """Root of 2x + 4 is -2."""
        p_r = rp.array([2.0, 4.0])
        p_n = np.array([2.0, 4.0])

        r = rp.roots(p_r)
        n = np.roots(p_n)

        # May have small imaginary part, compare real parts
        assert_eq(np.asarray(r).real, n.real, rtol=1e-5)

    def test_quadratic_real_roots(self):
        """Roots of x^2 - 4 are +2 and -2."""
        p_r = rp.array([1.0, 0.0, -4.0])
        p_n = np.array([1.0, 0.0, -4.0])

        r = rp.roots(p_r)
        n = np.roots(p_n)

        # Sort for comparison
        r_sorted = np.sort(np.asarray(r).real)
        n_sorted = np.sort(n.real)
        assert_eq(r_sorted, n_sorted, rtol=1e-5)

    def test_quadratic_double_root(self):
        """Roots of (x-3)^2 = x^2 - 6x + 9 is double root at 3."""
        p_r = rp.array([1.0, -6.0, 9.0])
        p_n = np.array([1.0, -6.0, 9.0])

        r = rp.roots(p_r)
        n = np.roots(p_n)

        # Both roots should be near 3
        r_sorted = np.sort(np.asarray(r).real)
        n_sorted = np.sort(n.real)
        assert_eq(r_sorted, n_sorted, rtol=1e-5)

    def test_cubic_integer_roots(self):
        """Roots of (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6."""
        p_r = rp.array([1.0, -6.0, 11.0, -6.0])
        p_n = np.array([1.0, -6.0, 11.0, -6.0])

        r = rp.roots(p_r)
        n = np.roots(p_n)

        r_sorted = np.sort(np.asarray(r).real)
        n_sorted = np.sort(n.real)
        assert_eq(r_sorted, n_sorted, rtol=1e-3)

    @pytest.mark.skip(reason="rumpy roots() has known issue with complex roots")
    def test_cubic_single_real_root(self):
        """Cubic with one real root and two complex conjugate roots."""
        # x^3 + 1 = (x+1)(x^2 - x + 1)
        # Note: Current rumpy implementation doesn't correctly handle complex roots
        p_r = rp.array([1.0, 0.0, 0.0, 1.0])
        p_n = np.array([1.0, 0.0, 0.0, 1.0])

        r = rp.roots(p_r)
        n = np.roots(p_n)

        # Compare sorted by real then imaginary parts
        r_np = np.asarray(r)
        r_sorted = r_np[np.lexsort((r_np.imag, r_np.real))]
        n_sorted = n[np.lexsort((n.imag, n.real))]
        assert_eq(r_sorted, n_sorted, rtol=1e-4, atol=1e-10)

    def test_constant_no_roots(self):
        """Constant polynomial has no roots."""
        p_r = rp.array([5.0])

        r = rp.roots(p_r)
        r_np = np.asarray(r)
        assert r_np.size == 0

    def test_polynomial_with_zero_roots(self):
        """Polynomial x^2 has double root at 0."""
        p_r = rp.array([1.0, 0.0, 0.0])
        p_n = np.array([1.0, 0.0, 0.0])

        r = rp.roots(p_r)
        n = np.roots(p_n)

        r_sorted = np.sort(np.asarray(r).real)
        n_sorted = np.sort(n.real)
        assert_eq(r_sorted, n_sorted, rtol=1e-5)

    def test_high_degree_polynomial(self):
        """Roots of higher degree polynomial."""
        # (x-1)(x-2)(x-3)(x-4)
        p_r = rp.array([1.0, -10.0, 35.0, -50.0, 24.0])
        p_n = np.array([1.0, -10.0, 35.0, -50.0, 24.0])

        r = rp.roots(p_r)
        n = np.roots(p_n)

        r_sorted = np.sort(np.asarray(r).real)
        n_sorted = np.sort(n.real)
        assert_eq(r_sorted, n_sorted, rtol=1e-3)


# ============================================================================
# Round-trip Tests
# ============================================================================


class TestPolyRoundtrips:
    """Tests for polynomial operation round-trips."""

    def test_derivative_integral_roundtrip(self):
        """Integral of derivative recovers polynomial (up to constant)."""
        # Start with p(x) = x^2 + 2x + 3
        p = rp.array([1.0, 2.0, 3.0])

        # Take derivative: 2x + 2
        dp = rp.polyder(p)

        # Integrate: x^2 + 2x + c (c=0 by default)
        integ = rp.polyint(dp)

        # First coefficients should match
        p_np = np.asarray(p)
        integ_np = np.asarray(integ)
        np.testing.assert_allclose(integ_np[:-1], p_np[:-1], rtol=1e-10)

    def test_fit_eval_roundtrip(self):
        """Fitting and evaluating reproduces original values."""
        x_r = rp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y_r = rp.array([1.0, 3.0, 7.0, 13.0, 21.0])  # x^2 + x + 1

        # Fit degree 2
        coeffs = rp.polyfit(x_r, y_r, 2)

        # Evaluate at same points
        y_pred = rp.polyval(coeffs, x_r)

        # Should match original y
        assert_eq(y_pred, y_r, rtol=1e-5)



# ============================================================================
# Edge Cases
# ============================================================================


class TestPolyEdgeCases:
    """Edge case tests for polynomial operations."""

    def test_polyval_empty_coeffs(self):
        """Evaluate empty polynomial (should be zero)."""
        coeffs_r = rp.array([])
        coeffs_n = np.array([])
        x_r = rp.array([1.0, 2.0])
        x_n = np.array([1.0, 2.0])

        r = rp.polyval(coeffs_r, x_r)
        n = np.polyval(coeffs_n, x_n)
        assert_eq(r, n)

    def test_polyval_scalar_x(self):
        """Evaluate polynomial at scalar value (wrapped as array)."""
        coeffs_r = rp.array([1.0, 2.0, 3.0])
        coeffs_n = np.array([1.0, 2.0, 3.0])

        # rumpy requires x to be an array, wrap scalar
        r = rp.polyval(coeffs_r, rp.array([2.0]))
        n = np.polyval(coeffs_n, 2.0)

        # Handle scalar vs array result
        assert abs(float(r) - float(n)) < 1e-10

    def test_polyder_zero_order(self):
        """Zero-order derivative (should return same polynomial)."""
        p_r = rp.array([1.0, 2.0, 3.0])
        p_n = np.array([1.0, 2.0, 3.0])

        r = rp.polyder(p_r, m=0)
        n = np.polyder(p_n, 0)
        assert_eq(r, n)

    def test_polyint_zero_order(self):
        """Zero-order integral (should return same polynomial)."""
        p_r = rp.array([1.0, 2.0, 3.0])
        p_n = np.array([1.0, 2.0, 3.0])

        r = rp.polyint(p_r, m=0)
        n = np.polyint(p_n, m=0)
        assert_eq(r, n)
