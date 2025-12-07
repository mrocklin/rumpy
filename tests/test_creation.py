"""Tests for array creation functions."""

import numpy as np
import pytest

import rumpy as rp
from helpers import assert_eq


class TestZeros:
    """Test rp.zeros against np.zeros."""

    def test_1d(self):
        r = rp.zeros(10)
        n = np.zeros(10)
        assert_eq(r, n)

    def test_2d(self):
        r = rp.zeros((3, 4))
        n = np.zeros((3, 4))
        assert_eq(r, n)

    def test_3d(self):
        r = rp.zeros((2, 3, 4))
        n = np.zeros((2, 3, 4))
        assert_eq(r, n)

    def test_dtype_float32(self):
        r = rp.zeros(10, dtype="float32")
        n = np.zeros(10, dtype=np.float32)
        assert_eq(r, n)

    def test_dtype_float64(self):
        r = rp.zeros(10, dtype="float64")
        n = np.zeros(10, dtype=np.float64)
        assert_eq(r, n)

    def test_dtype_int32(self):
        r = rp.zeros(10, dtype="int32")
        n = np.zeros(10, dtype=np.int32)
        assert_eq(r, n)

    def test_dtype_int64(self):
        r = rp.zeros(10, dtype="int64")
        n = np.zeros(10, dtype=np.int64)
        assert_eq(r, n)

    def test_dtype_bool(self):
        r = rp.zeros(10, dtype="bool")
        n = np.zeros(10, dtype=bool)
        assert_eq(r, n)


class TestOnes:
    """Test rp.ones against np.ones."""

    def test_1d(self):
        r = rp.ones(10)
        n = np.ones(10)
        assert_eq(r, n)

    def test_2d(self):
        r = rp.ones((3, 4))
        n = np.ones((3, 4))
        assert_eq(r, n)

    def test_dtype_float32(self):
        r = rp.ones(10, dtype="float32")
        n = np.ones(10, dtype=np.float32)
        assert_eq(r, n)

    def test_dtype_float64(self):
        r = rp.ones(10, dtype="float64")
        n = np.ones(10, dtype=np.float64)
        assert_eq(r, n)

    def test_dtype_int32(self):
        r = rp.ones(10, dtype="int32")
        n = np.ones(10, dtype=np.int32)
        assert_eq(r, n)

    def test_dtype_int64(self):
        r = rp.ones(10, dtype="int64")
        n = np.ones(10, dtype=np.int64)
        assert_eq(r, n)

    def test_dtype_bool(self):
        r = rp.ones(10, dtype="bool")
        n = np.ones(10, dtype=bool)
        assert_eq(r, n)


class TestArrayProperties:
    """Test array properties match numpy."""

    def test_shape(self):
        r = rp.zeros((3, 4, 5))
        assert r.shape == (3, 4, 5)

    def test_ndim(self):
        r = rp.zeros((3, 4, 5))
        assert r.ndim == 3

    def test_size(self):
        r = rp.zeros((3, 4, 5))
        assert r.size == 60

    def test_itemsize_float64(self):
        r = rp.zeros(10, dtype="float64")
        assert r.itemsize == 8

    def test_itemsize_float32(self):
        r = rp.zeros(10, dtype="float32")
        assert r.itemsize == 4

    def test_nbytes(self):
        r = rp.zeros(10, dtype="float64")
        assert r.nbytes == 80

    def test_strides_c_order(self):
        r = rp.zeros((3, 4))
        # C-order: last dimension has stride = itemsize
        assert r.strides == (32, 8)  # 4*8, 1*8 for float64


class TestEmpty:
    """Test rp.empty against np.empty."""

    def test_1d(self):
        r = rp.empty(10)
        n = np.empty(10)
        assert r.shape == n.shape
        assert r.dtype == "float64"

    def test_2d(self):
        r = rp.empty((3, 4))
        n = np.empty((3, 4))
        assert r.shape == n.shape

    def test_dtype(self):
        r = rp.empty(10, dtype="float32")
        assert r.dtype == "float32"


class TestZerosLike:
    """Test rp.zeros_like against np.zeros_like."""

    def test_basic(self):
        arr = rp.ones((3, 4))
        r = rp.zeros_like(arr)
        n = np.zeros_like(np.ones((3, 4)))
        assert_eq(r, n)

    def test_preserves_dtype(self):
        arr = rp.ones((3, 4), dtype="float32")
        r = rp.zeros_like(arr)
        assert r.dtype == "float32"
        assert r.shape == (3, 4)

    def test_override_dtype(self):
        arr = rp.ones((3, 4), dtype="float32")
        r = rp.zeros_like(arr, dtype="float64")
        assert r.dtype == "float64"


class TestOnesLike:
    """Test rp.ones_like against np.ones_like."""

    def test_basic(self):
        arr = rp.zeros((3, 4))
        r = rp.ones_like(arr)
        n = np.ones_like(np.zeros((3, 4)))
        assert_eq(r, n)

    def test_preserves_dtype(self):
        arr = rp.zeros((3, 4), dtype="int32")
        r = rp.ones_like(arr)
        assert r.dtype == "int32"
        assert r.shape == (3, 4)

    def test_override_dtype(self):
        arr = rp.zeros((3, 4), dtype="int32")
        r = rp.ones_like(arr, dtype="float64")
        assert r.dtype == "float64"


class TestEmptyLike:
    """Test rp.empty_like against np.empty_like."""

    def test_basic(self):
        arr = rp.ones((3, 4))
        r = rp.empty_like(arr)
        assert r.shape == (3, 4)
        assert r.dtype == "float64"

    def test_preserves_dtype(self):
        arr = rp.ones((3, 4), dtype="float32")
        r = rp.empty_like(arr)
        assert r.dtype == "float32"

    def test_override_dtype(self):
        arr = rp.ones((3, 4), dtype="float32")
        r = rp.empty_like(arr, dtype="int64")
        assert r.dtype == "int64"


class TestCopyModule:
    """Test rp.copy module function."""

    def test_basic(self):
        arr = rp.arange(10)
        r = rp.copy(arr)
        n = np.copy(np.arange(10, dtype=np.float64))
        assert_eq(r, n)

    def test_2d(self):
        arr = rp.arange(12).reshape(3, 4)
        r = rp.copy(arr)
        n = np.copy(np.arange(12, dtype=np.float64).reshape(3, 4))
        assert_eq(r, n)

    def test_preserves_dtype(self):
        arr = rp.arange(10, dtype="int32")
        r = rp.copy(arr)
        assert r.dtype == "int32"


class TestArrayConstructor:
    """Test rp.array() constructor."""

    def test_from_list_1d(self):
        r = rp.array([1, 2, 3, 4])
        n = np.array([1, 2, 3, 4], dtype=np.float64)
        assert_eq(r, n)

    def test_from_list_2d(self):
        r = rp.array([[1, 2], [3, 4]])
        n = np.array([[1, 2], [3, 4]], dtype=np.float64)
        assert_eq(r, n)

    def test_with_dtype(self):
        r = rp.array([1, 2, 3], dtype="int32")
        assert r.dtype == "int32"
        assert r.shape == (3,)

    def test_from_rumpy_array(self):
        orig = rp.arange(5)
        r = rp.array(orig)
        assert_eq(r, orig)


class TestDtypeConstants:
    """Test that dtype constants are available and work with array creation."""

    @pytest.mark.parametrize("attr,expected", [
        ("float32", "float32"),
        ("float64", "float64"),
        ("int16", "int16"),
        ("int32", "int32"),
        ("int64", "int64"),
        ("uint8", "uint8"),
        ("uint16", "uint16"),
        ("uint32", "uint32"),
        ("uint64", "uint64"),
        ("bool_", "bool"),
        ("complex64", "complex64"),
        ("complex128", "complex128"),
    ])
    def test_dtype_constant_with_zeros(self, attr, expected):
        """Dtype constants should work with array creation functions."""
        dtype_const = getattr(rp, attr)
        arr = rp.zeros(5, dtype=dtype_const)
        assert arr.dtype == expected


class TestFullLike:
    """Test rp.full_like against np.full_like."""

    def test_basic(self):
        arr = rp.zeros((3, 4))
        r = rp.full_like(arr, 7.0)
        n = np.full_like(np.zeros((3, 4)), 7.0)
        assert_eq(r, n)

    def test_preserves_dtype(self):
        arr = rp.zeros((3, 4), dtype="float32")
        r = rp.full_like(arr, 5.0)
        assert r.dtype == "float32"
        assert r.shape == (3, 4)

    def test_override_dtype(self):
        arr = rp.zeros((3, 4), dtype="float32")
        r = rp.full_like(arr, 5.0, dtype="float64")
        assert r.dtype == "float64"


class TestIdentity:
    """Test rp.identity against np.identity."""

    def test_basic(self):
        r = rp.identity(3)
        n = np.identity(3)
        assert_eq(r, n)

    def test_dtype(self):
        r = rp.identity(4, dtype="float32")
        n = np.identity(4, dtype=np.float32)
        assert_eq(r, n)

    def test_empty(self):
        r = rp.identity(0)
        n = np.identity(0)
        assert r.shape == n.shape


class TestLogspace:
    """Test rp.logspace against np.logspace."""

    def test_basic(self):
        r = rp.logspace(0, 2, num=5)
        n = np.logspace(0, 2, num=5)
        assert_eq(r, n)

    def test_with_base(self):
        r = rp.logspace(0, 3, num=4, base=2.0)
        n = np.logspace(0, 3, num=4, base=2.0)
        assert_eq(r, n)

    def test_single_point(self):
        r = rp.logspace(0, 2, num=1)
        n = np.logspace(0, 2, num=1)
        assert_eq(r, n)

    def test_empty(self):
        r = rp.logspace(0, 2, num=0)
        n = np.logspace(0, 2, num=0)
        assert r.shape == n.shape


class TestGeomspace:
    """Test rp.geomspace against np.geomspace."""

    def test_basic(self):
        r = rp.geomspace(1, 1000, num=4)
        n = np.geomspace(1, 1000, num=4)
        assert_eq(r, n)

    def test_negative(self):
        r = rp.geomspace(-1, -1000, num=4)
        n = np.geomspace(-1, -1000, num=4)
        assert_eq(r, n)

    def test_single_point(self):
        r = rp.geomspace(1, 100, num=1)
        n = np.geomspace(1, 100, num=1)
        assert_eq(r, n)

    def test_invalid_signs(self):
        with pytest.raises(ValueError):
            rp.geomspace(-1, 100, num=4)


class TestTri:
    """Test rp.tri against np.tri."""

    def test_basic(self):
        r = rp.tri(3)
        n = np.tri(3)
        assert_eq(r, n)

    def test_rectangular(self):
        r = rp.tri(3, 4)
        n = np.tri(3, 4)
        assert_eq(r, n)

    def test_with_k(self):
        r = rp.tri(3, 4, k=1)
        n = np.tri(3, 4, k=1)
        assert_eq(r, n)

    def test_with_negative_k(self):
        r = rp.tri(4, 4, k=-1)
        n = np.tri(4, 4, k=-1)
        assert_eq(r, n)


class TestTril:
    """Test rp.tril against np.tril."""

    def test_basic(self):
        arr = rp.arange(12).reshape(3, 4)
        r = rp.tril(arr)
        n = np.tril(np.arange(12, dtype=np.float64).reshape(3, 4))
        assert_eq(r, n)

    def test_with_k(self):
        arr = rp.arange(12).reshape(3, 4)
        r = rp.tril(arr, k=1)
        n = np.tril(np.arange(12, dtype=np.float64).reshape(3, 4), k=1)
        assert_eq(r, n)

    def test_with_negative_k(self):
        arr = rp.arange(16).reshape(4, 4)
        r = rp.tril(arr, k=-1)
        n = np.tril(np.arange(16, dtype=np.float64).reshape(4, 4), k=-1)
        assert_eq(r, n)


class TestTriu:
    """Test rp.triu against np.triu."""

    def test_basic(self):
        arr = rp.arange(12).reshape(3, 4)
        r = rp.triu(arr)
        n = np.triu(np.arange(12, dtype=np.float64).reshape(3, 4))
        assert_eq(r, n)

    def test_with_k(self):
        arr = rp.arange(12).reshape(3, 4)
        r = rp.triu(arr, k=1)
        n = np.triu(np.arange(12, dtype=np.float64).reshape(3, 4), k=1)
        assert_eq(r, n)

    def test_with_negative_k(self):
        arr = rp.arange(16).reshape(4, 4)
        r = rp.triu(arr, k=-1)
        n = np.triu(np.arange(16, dtype=np.float64).reshape(4, 4), k=-1)
        assert_eq(r, n)


class TestDiagflat:
    """Test rp.diagflat against np.diagflat."""

    def test_basic(self):
        arr = rp.array([1, 2, 3])
        r = rp.diagflat(arr)
        n = np.diagflat(np.array([1, 2, 3], dtype=np.float64))
        assert_eq(r, n)

    def test_with_positive_k(self):
        arr = rp.array([1, 2, 3])
        r = rp.diagflat(arr, k=1)
        n = np.diagflat(np.array([1, 2, 3], dtype=np.float64), k=1)
        assert_eq(r, n)

    def test_with_negative_k(self):
        arr = rp.array([1, 2, 3])
        r = rp.diagflat(arr, k=-1)
        n = np.diagflat(np.array([1, 2, 3], dtype=np.float64), k=-1)
        assert_eq(r, n)

    def test_2d_input(self):
        """diagflat flattens input first."""
        arr = rp.arange(4).reshape(2, 2)
        r = rp.diagflat(arr)
        n = np.diagflat(np.arange(4, dtype=np.float64).reshape(2, 2))
        assert_eq(r, n)


class TestMeshgrid:
    """Test rp.meshgrid against np.meshgrid."""

    def test_basic_xy(self):
        x = rp.array([1, 2, 3])
        y = rp.array([4, 5])
        rx, ry = rp.meshgrid(x, y)
        nx, ny = np.meshgrid(np.array([1, 2, 3], dtype=np.float64),
                              np.array([4, 5], dtype=np.float64))
        assert_eq(rx, nx)
        assert_eq(ry, ny)

    def test_indexing_ij(self):
        x = rp.array([1, 2, 3])
        y = rp.array([4, 5])
        rx, ry = rp.meshgrid(x, y, indexing="ij")
        nx, ny = np.meshgrid(np.array([1, 2, 3], dtype=np.float64),
                              np.array([4, 5], dtype=np.float64),
                              indexing="ij")
        assert_eq(rx, nx)
        assert_eq(ry, ny)

    def test_3d(self):
        x = rp.array([1, 2])
        y = rp.array([3, 4, 5])
        z = rp.array([6, 7])
        rx, ry, rz = rp.meshgrid(x, y, z, indexing="ij")
        nx, ny, nz = np.meshgrid(np.array([1, 2], dtype=np.float64),
                                  np.array([3, 4, 5], dtype=np.float64),
                                  np.array([6, 7], dtype=np.float64),
                                  indexing="ij")
        assert_eq(rx, nx)
        assert_eq(ry, ny)
        assert_eq(rz, nz)


class TestIndices:
    """Test rp.indices against np.indices."""

    def test_basic_2d(self):
        r = rp.indices([2, 3])
        n = np.indices([2, 3])
        assert_eq(r, n)

    def test_3d(self):
        r = rp.indices([2, 3, 4])
        n = np.indices([2, 3, 4])
        assert_eq(r, n)

    def test_1d(self):
        r = rp.indices([5])
        n = np.indices([5])
        assert_eq(r, n)


class TestFromfunction:
    """Test rp.fromfunction against np.fromfunction."""

    def test_basic_2d(self):
        def f(i, j):
            return i + j
        r = rp.fromfunction([3, 4], f)
        n = np.fromfunction(f, (3, 4))
        assert_eq(r, n)

    def test_multiply(self):
        def f(i, j):
            return i * j
        r = rp.fromfunction([3, 3], f)
        n = np.fromfunction(f, (3, 3))
        assert_eq(r, n)

    def test_3d(self):
        def f(i, j, k):
            return i + j + k
        r = rp.fromfunction([2, 3, 2], f)
        n = np.fromfunction(f, (2, 3, 2))
        assert_eq(r, n)
