"""Tests for array inspection functions (Stream 27)."""
import numpy as np
import pytest
import rumpy as rp
from helpers import assert_eq


class TestIsneginf:
    """Tests for isneginf."""

    def test_basic(self):
        n = np.array([1.0, -np.inf, np.inf, np.nan, 0.0, -1.0])
        r = rp.asarray(n)
        assert_eq(rp.isneginf(r), np.isneginf(n))

    def test_int_array(self):
        # Integers can't be inf
        n = np.array([1, 2, 3, -1, 0])
        r = rp.asarray(n)
        assert_eq(rp.isneginf(r), np.isneginf(n))

    def test_scalar(self):
        assert rp.isneginf(-np.inf) == np.isneginf(-np.inf)
        assert rp.isneginf(np.inf) == np.isneginf(np.inf)
        assert rp.isneginf(1.0) == np.isneginf(1.0)


class TestIsposinf:
    """Tests for isposinf."""

    def test_basic(self):
        n = np.array([1.0, -np.inf, np.inf, np.nan, 0.0, -1.0])
        r = rp.asarray(n)
        assert_eq(rp.isposinf(r), np.isposinf(n))

    def test_int_array(self):
        n = np.array([1, 2, 3, -1, 0])
        r = rp.asarray(n)
        assert_eq(rp.isposinf(r), np.isposinf(n))

    def test_scalar(self):
        assert rp.isposinf(np.inf) == np.isposinf(np.inf)
        assert rp.isposinf(-np.inf) == np.isposinf(-np.inf)
        assert rp.isposinf(1.0) == np.isposinf(1.0)


class TestIsreal:
    """Tests for isreal (element-wise)."""

    def test_real_array(self):
        n = np.array([1.0, 2.0, 3.0])
        r = rp.asarray(n)
        assert_eq(rp.isreal(r), np.isreal(n))

    def test_complex_array(self):
        n = np.array([1+0j, 2+1j, 3+0j])
        r = rp.asarray(n)
        assert_eq(rp.isreal(r), np.isreal(n))

    def test_scalar(self):
        assert rp.isreal(1.0) == np.isreal(1.0)
        # Complex scalars not supported in rumpy (only arrays)


class TestIscomplex:
    """Tests for iscomplex (element-wise)."""

    def test_real_array(self):
        n = np.array([1.0, 2.0, 3.0])
        r = rp.asarray(n)
        assert_eq(rp.iscomplex(r), np.iscomplex(n))

    def test_complex_array(self):
        n = np.array([1+0j, 2+1j, 3+0j])
        r = rp.asarray(n)
        assert_eq(rp.iscomplex(r), np.iscomplex(n))


class TestIsrealobj:
    """Tests for isrealobj (checks dtype, not values)."""

    def test_real_dtypes(self):
        for dtype in [np.float32, np.float64, np.int32, np.int64]:
            n = np.array([1, 2, 3], dtype=dtype)
            r = rp.asarray(n)
            assert rp.isrealobj(r) == np.isrealobj(n)

    def test_complex_dtypes(self):
        for dtype in [np.complex64, np.complex128]:
            n = np.array([1+0j, 2+0j], dtype=dtype)
            r = rp.asarray(n)
            assert rp.isrealobj(r) == np.isrealobj(n)


class TestIscomplexobj:
    """Tests for iscomplexobj (checks dtype, not values)."""

    def test_real_dtypes(self):
        for dtype in [np.float32, np.float64, np.int32, np.int64]:
            n = np.array([1, 2, 3], dtype=dtype)
            r = rp.asarray(n)
            assert rp.iscomplexobj(r) == np.iscomplexobj(n)

    def test_complex_dtypes(self):
        for dtype in [np.complex64, np.complex128]:
            n = np.array([1+0j, 2+0j], dtype=dtype)
            r = rp.asarray(n)
            assert rp.iscomplexobj(r) == np.iscomplexobj(n)


class TestSharesMemory:
    """Tests for shares_memory."""

    def test_view_shares(self):
        n = np.array([1, 2, 3, 4, 5])
        r = rp.asarray(n)
        r_view = r[1:]
        assert rp.shares_memory(r, r_view) == True

    def test_copy_does_not_share(self):
        n = np.array([1, 2, 3])
        r1 = rp.asarray(n)
        r2 = rp.asarray(n)  # separate array
        assert rp.shares_memory(r1, r2) == False

    def test_same_array_shares(self):
        r = rp.asarray([1, 2, 3])
        assert rp.shares_memory(r, r) == True


class TestMayShareMemory:
    """Tests for may_share_memory."""

    def test_view_may_share(self):
        r = rp.asarray([1, 2, 3, 4, 5])
        r_view = r[1:]
        assert rp.may_share_memory(r, r_view) == True

    def test_separate_does_not_share(self):
        r1 = rp.asarray([1, 2, 3])
        r2 = rp.asarray([1, 2, 3])
        assert rp.may_share_memory(r1, r2) == False


class TestIsscalar:
    """Tests for isscalar."""

    def test_python_scalars(self):
        assert rp.isscalar(1) == np.isscalar(1)
        assert rp.isscalar(1.0) == np.isscalar(1.0)
        assert rp.isscalar(1+1j) == np.isscalar(1+1j)
        assert rp.isscalar("hello") == np.isscalar("hello")
        assert rp.isscalar(True) == np.isscalar(True)

    def test_non_scalars(self):
        assert rp.isscalar([1]) == np.isscalar([1])
        assert rp.isscalar((1,)) == np.isscalar((1,))
        assert rp.isscalar(np.array(1)) == np.isscalar(np.array(1))
        assert rp.isscalar(rp.asarray([1])) == False  # array is not scalar


class TestModuleLevelNdim:
    """Tests for module-level ndim."""

    def test_array(self):
        for shape in [(), (3,), (2, 3), (2, 3, 4)]:
            n = np.zeros(shape)
            r = rp.zeros(shape)
            assert rp.ndim(r) == np.ndim(n)

    def test_list(self):
        assert rp.ndim([1, 2, 3]) == np.ndim([1, 2, 3])
        assert rp.ndim([[1, 2], [3, 4]]) == np.ndim([[1, 2], [3, 4]])

    def test_scalar(self):
        # rp.ndim doesn't support scalars (0-d arrays not supported)
        pass


class TestModuleLevelSize:
    """Tests for module-level size."""

    def test_array(self):
        for shape in [(), (3,), (2, 3), (2, 3, 4)]:
            n = np.zeros(shape)
            r = rp.zeros(shape)
            assert rp.size(r) == np.size(n)

    def test_list(self):
        assert rp.size([1, 2, 3]) == np.size([1, 2, 3])
        assert rp.size([[1, 2], [3, 4]]) == np.size([[1, 2], [3, 4]])


class TestModuleLevelShape:
    """Tests for module-level shape."""

    def test_array(self):
        for shape in [(), (3,), (2, 3), (2, 3, 4)]:
            n = np.zeros(shape)
            r = rp.zeros(shape)
            assert rp.shape(r) == np.shape(n)

    def test_list(self):
        assert rp.shape([1, 2, 3]) == np.shape([1, 2, 3])
        assert rp.shape([[1, 2], [3, 4]]) == np.shape([[1, 2], [3, 4]])
