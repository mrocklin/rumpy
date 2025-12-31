"""Tests for array creation functions.

Creation functions are simple wrappers, so we use CORE_DTYPES.
See designs/testing.md for testing philosophy.
"""

import numpy as np
import pytest

import rumpy as rp
from conftest import CORE_DTYPES, CORE_SHAPES, NUMERIC_DTYPES, SHAPES_EMPTY
from helpers import assert_eq


class TestZeros:
    """Test rp.zeros against np.zeros."""

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        r = rp.zeros(shape)
        n = np.zeros(shape)
        assert_eq(r, n)

    @pytest.mark.parametrize("dtype", CORE_DTYPES)
    def test_dtypes(self, dtype):
        r = rp.zeros(10, dtype=dtype)
        n = np.zeros(10, dtype=dtype)
        assert_eq(r, n)
        assert r.dtype == dtype

    @pytest.mark.parametrize("shape", SHAPES_EMPTY)
    def test_empty(self, shape):
        r = rp.zeros(shape)
        n = np.zeros(shape)
        assert r.shape == n.shape
        assert r.size == 0


class TestOnes:
    """Test rp.ones against np.ones."""

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        r = rp.ones(shape)
        n = np.ones(shape)
        assert_eq(r, n)

    @pytest.mark.parametrize("dtype", CORE_DTYPES)
    def test_dtypes(self, dtype):
        r = rp.ones(10, dtype=dtype)
        n = np.ones(10, dtype=dtype)
        assert_eq(r, n)
        assert r.dtype == dtype

    @pytest.mark.parametrize("shape", SHAPES_EMPTY)
    def test_empty(self, shape):
        r = rp.ones(shape)
        n = np.ones(shape)
        assert r.shape == n.shape


class TestEmpty:
    """Test rp.empty against np.empty."""

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        r = rp.empty(shape)
        n = np.empty(shape)
        assert r.shape == n.shape

    @pytest.mark.parametrize("dtype", CORE_DTYPES)
    def test_dtypes(self, dtype):
        r = rp.empty(10, dtype=dtype)
        assert r.dtype == dtype

    @pytest.mark.parametrize("shape", SHAPES_EMPTY)
    def test_empty(self, shape):
        r = rp.empty(shape)
        assert r.shape == shape
        assert r.size == 0


class TestFull:
    """Test rp.full against np.full."""

    @pytest.mark.parametrize("shape", CORE_SHAPES)
    def test_shapes(self, shape):
        r = rp.full(shape, 3.14)
        n = np.full(shape, 3.14)
        assert_eq(r, n)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_dtypes(self, dtype):
        r = rp.full(10, 42, dtype=dtype)
        n = np.full(10, 42, dtype=dtype)
        assert_eq(r, n)
        assert r.dtype == dtype

    def test_negative_value(self):
        r = rp.full((3, 3), -42.0)
        n = np.full((3, 3), -42.0)
        assert_eq(r, n)

    @pytest.mark.parametrize("shape", SHAPES_EMPTY)
    def test_empty(self, shape):
        r = rp.full(shape, 5.0)
        n = np.full(shape, 5.0)
        assert r.shape == n.shape


class TestArange:
    """Test rp.arange against np.arange."""

    def test_stop_only(self):
        r = rp.arange(10)
        n = np.arange(10)
        assert_eq(r, n)

    def test_start_stop(self):
        r = rp.arange(2, 10)
        n = np.arange(2, 10)
        assert_eq(r, n)

    def test_start_stop_step(self):
        r = rp.arange(0, 10, 2)
        n = np.arange(0, 10, 2)
        assert_eq(r, n)

    def test_float_step(self):
        r = rp.arange(0, 1, 0.1, dtype="float64")
        n = np.arange(0, 1, 0.1)
        assert_eq(r, n)

    def test_negative_step(self):
        r = rp.arange(10, 0, -1)
        n = np.arange(10, 0, -1)
        assert_eq(r, n)

    @pytest.mark.parametrize("dtype", ["int64", "float32", "float64"])
    def test_dtypes(self, dtype):
        r = rp.arange(10, dtype=dtype)
        n = np.arange(10, dtype=dtype)
        assert_eq(r, n)
        assert r.dtype == dtype

    def test_empty(self):
        r = rp.arange(0)
        n = np.arange(0)
        assert r.shape == n.shape


class TestLinspace:
    """Test rp.linspace against np.linspace."""

    def test_basic(self):
        r = rp.linspace(0, 10, 5)
        n = np.linspace(0, 10, 5)
        assert_eq(r, n)

    def test_default_num(self):
        r = rp.linspace(0, 1)
        n = np.linspace(0, 1)  # default 50 points
        assert_eq(r, n)

    def test_single_point(self):
        r = rp.linspace(5, 5, 1)
        n = np.linspace(5, 5, 1)
        assert_eq(r, n)

    def test_negative(self):
        r = rp.linspace(-5, 5, 11)
        n = np.linspace(-5, 5, 11)
        assert_eq(r, n)

    def test_reverse(self):
        r = rp.linspace(10, 0, 5)
        n = np.linspace(10, 0, 5)
        assert_eq(r, n)

    def test_empty(self):
        r = rp.linspace(0, 10, 0)
        n = np.linspace(0, 10, 0)
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


class TestEye:
    """Test rp.eye against np.eye."""

    def test_basic(self):
        r = rp.eye(3)
        n = np.eye(3)
        assert_eq(r, n)

    def test_sizes(self):
        for size in [1, 5, 10]:
            r = rp.eye(size)
            n = np.eye(size)
            assert_eq(r, n)

    @pytest.mark.parametrize("dtype", CORE_DTYPES)
    def test_dtypes(self, dtype):
        r = rp.eye(3, dtype=dtype)
        n = np.eye(3, dtype=dtype)
        assert_eq(r, n)
        assert r.dtype == dtype

    def test_empty(self):
        r = rp.eye(0)
        n = np.eye(0)
        assert r.shape == n.shape


class TestIdentity:
    """Test rp.identity against np.identity."""

    def test_basic(self):
        r = rp.identity(3)
        n = np.identity(3)
        assert_eq(r, n)

    @pytest.mark.parametrize("dtype", CORE_DTYPES)
    def test_dtypes(self, dtype):
        r = rp.identity(4, dtype=dtype)
        n = np.identity(4, dtype=dtype)
        assert_eq(r, n)

    def test_empty(self):
        r = rp.identity(0)
        n = np.identity(0)
        assert r.shape == n.shape


class TestDiag:
    """Test rp.diag against np.diag."""

    def test_extract_diagonal(self):
        # Note: must use float input, int diag has known bug
        n = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        r = rp.asarray(n)
        assert_eq(rp.diag(r), np.diag(n))

    def test_create_diagonal(self):
        n = np.array([1.0, 2.0, 3.0])
        r = rp.asarray(n)
        assert_eq(rp.diag(r), np.diag(n))

    def test_rectangular(self):
        n = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        r = rp.asarray(n)
        assert_eq(rp.diag(r), np.diag(n))


class TestDiagflat:
    """Test rp.diagflat against np.diagflat."""

    def test_basic(self):
        arr = rp.array([1, 2, 3])
        n = np.array([1, 2, 3], dtype=np.float64)
        assert_eq(rp.diagflat(arr), np.diagflat(n))

    def test_with_k(self):
        arr = rp.array([1, 2, 3])
        n = np.array([1, 2, 3], dtype=np.float64)
        for k in [-1, 0, 1]:
            assert_eq(rp.diagflat(arr, k=k), np.diagflat(n, k=k))

    def test_2d_input(self):
        arr = rp.arange(4).reshape(2, 2)
        n = np.arange(4, dtype=np.float64).reshape(2, 2)
        assert_eq(rp.diagflat(arr), np.diagflat(n))


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

    @pytest.mark.parametrize("k", [-1, 0, 1])
    def test_with_k(self, k):
        r = rp.tri(4, 4, k=k)
        n = np.tri(4, 4, k=k)
        assert_eq(r, n)


class TestTrilTriu:
    """Test rp.tril and rp.triu against numpy."""

    def test_tril_basic(self):
        arr = rp.arange(12).reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(rp.tril(arr), np.tril(n))

    def test_triu_basic(self):
        arr = rp.arange(12).reshape(3, 4)
        n = np.arange(12, dtype=np.float64).reshape(3, 4)
        assert_eq(rp.triu(arr), np.triu(n))

    @pytest.mark.parametrize("k", [-1, 0, 1])
    def test_tril_with_k(self, k):
        arr = rp.arange(16).reshape(4, 4)
        n = np.arange(16, dtype=np.float64).reshape(4, 4)
        assert_eq(rp.tril(arr, k=k), np.tril(n, k=k))

    @pytest.mark.parametrize("k", [-1, 0, 1])
    def test_triu_with_k(self, k):
        arr = rp.arange(16).reshape(4, 4)
        n = np.arange(16, dtype=np.float64).reshape(4, 4)
        assert_eq(rp.triu(arr, k=k), np.triu(n, k=k))


class TestMeshgrid:
    """Test rp.meshgrid against np.meshgrid."""

    def test_basic_xy(self):
        x, y = rp.array([1, 2, 3]), rp.array([4, 5])
        nx, ny = np.array([1, 2, 3], dtype=np.float64), np.array([4, 5], dtype=np.float64)
        rx, ry = rp.meshgrid(x, y)
        enx, eny = np.meshgrid(nx, ny)
        assert_eq(rx, enx)
        assert_eq(ry, eny)

    def test_indexing_ij(self):
        x, y = rp.array([1, 2, 3]), rp.array([4, 5])
        nx, ny = np.array([1, 2, 3], dtype=np.float64), np.array([4, 5], dtype=np.float64)
        rx, ry = rp.meshgrid(x, y, indexing="ij")
        enx, eny = np.meshgrid(nx, ny, indexing="ij")
        assert_eq(rx, enx)
        assert_eq(ry, eny)

    def test_3d(self):
        x, y, z = rp.array([1, 2]), rp.array([3, 4, 5]), rp.array([6, 7])
        nx = np.array([1, 2], dtype=np.float64)
        ny = np.array([3, 4, 5], dtype=np.float64)
        nz = np.array([6, 7], dtype=np.float64)
        rx, ry, rz = rp.meshgrid(x, y, z, indexing="ij")
        enx, eny, enz = np.meshgrid(nx, ny, nz, indexing="ij")
        assert_eq(rx, enx)
        assert_eq(ry, eny)
        assert_eq(rz, enz)


class TestIndices:
    """Test rp.indices against np.indices."""

    def test_2d(self):
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

    def test_add(self):
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


class TestLikeFunctions:
    """Test zeros_like, ones_like, empty_like, full_like."""

    def test_zeros_like(self):
        arr = rp.ones((3, 4))
        r = rp.zeros_like(arr)
        n = np.zeros_like(np.ones((3, 4)))
        assert_eq(r, n)

    def test_ones_like(self):
        arr = rp.zeros((3, 4))
        r = rp.ones_like(arr)
        n = np.ones_like(np.zeros((3, 4)))
        assert_eq(r, n)

    def test_empty_like(self):
        arr = rp.ones((3, 4))
        r = rp.empty_like(arr)
        assert r.shape == (3, 4)
        assert r.dtype == "float64"

    def test_full_like(self):
        arr = rp.zeros((3, 4))
        r = rp.full_like(arr, 7.0)
        n = np.full_like(np.zeros((3, 4)), 7.0)
        assert_eq(r, n)

    @pytest.mark.parametrize("dtype", CORE_DTYPES)
    def test_preserves_dtype(self, dtype):
        arr = rp.ones((3, 4), dtype=dtype)
        r = rp.zeros_like(arr)
        assert r.dtype == dtype

    def test_override_dtype(self):
        arr = rp.ones((3, 4), dtype="float32")
        r = rp.zeros_like(arr, dtype="float64")
        assert r.dtype == "float64"


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

    @pytest.mark.parametrize("dtype", CORE_DTYPES)
    def test_with_dtype(self, dtype):
        r = rp.array([1, 2, 3], dtype=dtype)
        assert r.dtype == dtype

    def test_from_rumpy_array(self):
        orig = rp.arange(5)
        r = rp.array(orig)
        assert_eq(r, orig)


class TestCopy:
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

    @pytest.mark.parametrize("dtype", CORE_DTYPES)
    def test_preserves_dtype(self, dtype):
        arr = rp.arange(10, dtype=dtype if dtype != "bool" else "int64")
        r = rp.copy(arr)
        assert r.dtype == arr.dtype


class TestAsarray:
    """Test rp.asarray conversion."""

    def test_from_numpy(self):
        n = np.array([1, 2, 3], dtype=np.float64)
        r = rp.asarray(n)
        assert_eq(r, n)

    def test_from_list(self):
        r = rp.asarray([1, 2, 3])
        n = np.asarray([1, 2, 3], dtype=np.float64)
        assert_eq(r, n)

    def test_from_rumpy(self):
        orig = rp.arange(5)
        r = rp.asarray(orig)
        assert_eq(r, orig)


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


class TestDtypeConstants:
    """Test dtype constants work with array creation."""

    @pytest.mark.parametrize(
        "attr,expected",
        [
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
        ],
    )
    def test_dtype_constant_with_zeros(self, attr, expected):
        dtype_const = getattr(rp, attr)
        arr = rp.zeros(5, dtype=dtype_const)
        assert arr.dtype == expected


class TestNumpyDtypeObjects:
    """Test that numpy dtype objects work as dtype arguments."""

    @pytest.mark.parametrize(
        "np_dtype,expected",
        [
            (np.float32, "float32"),
            (np.float64, "float64"),
            (np.int32, "int32"),
            (np.int64, "int64"),
            (np.complex128, "complex128"),
        ],
    )
    def test_numpy_type_with_zeros(self, np_dtype, expected):
        """Test np.float64 style type objects."""
        arr = rp.zeros(5, dtype=np_dtype)
        assert arr.dtype == expected

    @pytest.mark.parametrize(
        "dtype_str",
        ["float32", "float64", "int32", "int64"],
    )
    def test_numpy_dtype_instance(self, dtype_str):
        """Test np.dtype('float64') style instances."""
        arr = rp.zeros(5, dtype=np.dtype(dtype_str))
        assert arr.dtype == dtype_str

    def test_numpy_dtype_with_arange(self):
        """Test numpy dtype with arange."""
        r = rp.arange(10, dtype=np.float32)
        n = np.arange(10, dtype=np.float32)
        assert_eq(r, n)

    def test_numpy_dtype_with_linspace(self):
        """Test numpy dtype with linspace."""
        r = rp.linspace(0, 1, 5, dtype=np.float32)
        n = np.linspace(0, 1, 5, dtype=np.float32)
        assert_eq(r, n)
