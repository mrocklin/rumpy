"""Tests for math ufuncs."""

import numpy as np

import rumpy
from helpers import assert_eq


class TestSqrt:
    """Test sqrt ufunc."""

    def test_sqrt_1d(self):
        r = rumpy.sqrt(rumpy.arange(0, 10))
        n = np.sqrt(np.arange(10, dtype=np.float64))
        assert_eq(r, n)

    def test_sqrt_2d(self):
        r = rumpy.sqrt(rumpy.arange(1, 13).reshape([3, 4]))
        n = np.sqrt(np.arange(1, 13, dtype=np.float64).reshape(3, 4))
        assert_eq(r, n)


class TestExp:
    """Test exp ufunc."""

    def test_exp_1d(self):
        r = rumpy.exp(rumpy.arange(0, 5))
        n = np.exp(np.arange(5, dtype=np.float64))
        assert_eq(r, n)

    def test_exp_negative(self):
        r = rumpy.exp(rumpy.arange(-3, 3))
        n = np.exp(np.arange(-3, 3, dtype=np.float64))
        assert_eq(r, n)

    def test_exp_2d(self):
        r = rumpy.exp(rumpy.arange(6).reshape([2, 3]))
        n = np.exp(np.arange(6, dtype=np.float64).reshape(2, 3))
        assert_eq(r, n)


class TestLog:
    """Test log (natural logarithm) ufunc."""

    def test_log_1d(self):
        r = rumpy.log(rumpy.arange(1, 10))
        n = np.log(np.arange(1, 10, dtype=np.float64))
        assert_eq(r, n)

    def test_log_2d(self):
        r = rumpy.log(rumpy.arange(1, 7).reshape([2, 3]))
        n = np.log(np.arange(1, 7, dtype=np.float64).reshape(2, 3))
        assert_eq(r, n)

    def test_log_exp_inverse(self):
        x = rumpy.arange(1, 5)
        assert_eq(rumpy.exp(rumpy.log(x)), np.arange(1, 5, dtype=np.float64))


class TestSin:
    """Test sin ufunc."""

    def test_sin_1d(self):
        r = rumpy.sin(rumpy.linspace(0, 6.28, 10))
        n = np.sin(np.linspace(0, 6.28, 10))
        assert_eq(r, n)

    def test_sin_2d(self):
        r = rumpy.sin(rumpy.linspace(0, 3.14, 6).reshape([2, 3]))
        n = np.sin(np.linspace(0, 3.14, 6).reshape(2, 3))
        assert_eq(r, n)


class TestCos:
    """Test cos ufunc."""

    def test_cos_1d(self):
        r = rumpy.cos(rumpy.linspace(0, 6.28, 10))
        n = np.cos(np.linspace(0, 6.28, 10))
        assert_eq(r, n)

    def test_cos_2d(self):
        r = rumpy.cos(rumpy.linspace(0, 3.14, 6).reshape([2, 3]))
        n = np.cos(np.linspace(0, 3.14, 6).reshape(2, 3))
        assert_eq(r, n)


class TestTan:
    """Test tan ufunc."""

    def test_tan_1d(self):
        # Avoid pi/2 where tan is undefined
        r = rumpy.tan(rumpy.linspace(0, 1, 10))
        n = np.tan(np.linspace(0, 1, 10))
        assert_eq(r, n)

    def test_tan_2d(self):
        r = rumpy.tan(rumpy.linspace(0, 1, 6).reshape([2, 3]))
        n = np.tan(np.linspace(0, 1, 6).reshape(2, 3))
        assert_eq(r, n)


class TestSinCosPythagorean:
    """Test sin^2 + cos^2 = 1."""

    def test_pythagorean_identity(self):
        x = rumpy.linspace(0, 6.28, 100)
        s = rumpy.sin(x)
        c = rumpy.cos(x)
        result = s * s + c * c
        expected = rumpy.ones([100])
        assert_eq(result, np.asarray(expected))
