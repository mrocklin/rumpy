"""Tests for the C-API compatibility layer."""

import rumpy as rp


class TestCapiModuleAttributes:
    """Test that C-API module attributes are exported."""

    def test_array_api_version(self):
        """Test that __array_api_version__ is exported."""
        assert hasattr(rp, "__array_api_version__")
        assert rp.__array_api_version__ == "2.0"

    def test_array_api_exists(self):
        """Test that _ARRAY_API_EXISTS marker is exported."""
        assert hasattr(rp, "_ARRAY_API_EXISTS")
        assert rp._ARRAY_API_EXISTS is True


class TestArrayInterface:
    """Test that __array_interface__ provides C-compatible info."""

    def test_array_interface_exists(self):
        """Test that arrays have __array_interface__."""
        arr = rp.zeros((3, 4), dtype="float64")
        assert hasattr(arr, "__array_interface__")

    def test_array_interface_shape(self):
        """Test shape in array interface."""
        arr = rp.zeros((3, 4), dtype="float64")
        info = arr.__array_interface__
        assert info["shape"] == (3, 4)

    def test_array_interface_strides(self):
        """Test strides in array interface."""
        arr = rp.zeros((3, 4), dtype="float64")
        info = arr.__array_interface__
        # C-contiguous float64: stride for last dim = 8, first dim = 4*8=32
        assert info["strides"] == (32, 8)

    def test_array_interface_data(self):
        """Test data pointer in array interface."""
        arr = rp.zeros((3, 4), dtype="float64")
        info = arr.__array_interface__
        # data is a tuple (ptr, readonly)
        assert "data" in info
        data = info["data"]
        assert isinstance(data, tuple)
        assert len(data) == 2
        assert isinstance(data[0], int)  # Pointer
        assert isinstance(data[1], bool)  # Read-only flag

    def test_array_interface_typestr(self):
        """Test typestr in array interface."""
        arr = rp.zeros((3,), dtype="float64")
        info = arr.__array_interface__
        # float64 on little-endian is '<f8'
        assert info["typestr"] in ("<f8", ">f8", "=f8")

    def test_array_interface_dtype_int32(self):
        """Test typestr for int32."""
        arr = rp.zeros((3,), dtype="int32")
        info = arr.__array_interface__
        assert info["typestr"] in ("<i4", ">i4", "=i4")

    def test_array_interface_dtype_bool(self):
        """Test typestr for bool."""
        arr = rp.zeros((3,), dtype="bool")
        info = arr.__array_interface__
        assert info["typestr"] == "|b1"  # Bool is single byte, no endian

    def test_array_interface_version(self):
        """Test interface version."""
        arr = rp.zeros((3,), dtype="float64")
        info = arr.__array_interface__
        assert info["version"] == 3


class TestNumpyInterop:
    """Test interoperability with NumPy via array interface."""

    def test_numpy_asarray_uses_interface(self):
        """Test that NumPy can create array from rumpy via interface."""
        import numpy as np

        r = rp.zeros((3, 4), dtype="float64")
        n = np.asarray(r)

        assert n.shape == (3, 4)
        assert n.dtype == np.float64

    def test_numpy_views_rumpy_data(self):
        """Test that NumPy views the same data (zero-copy when possible)."""
        import numpy as np

        r = rp.arange(10, dtype="float64")
        n = np.asarray(r)

        # Check they point to the same memory
        r_ptr = r.__array_interface__["data"][0]
        n_ptr = n.__array_interface__["data"][0]
        assert r_ptr == n_ptr

    def test_rumpy_from_numpy(self):
        """Test that rumpy can create array from NumPy."""
        import numpy as np

        n = np.arange(10, dtype=np.float64)
        r = rp.asarray(n)

        # Values should match
        n2 = np.asarray(r)
        assert np.allclose(n, n2)
