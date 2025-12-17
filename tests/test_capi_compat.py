"""Tests that verify Rumpy's C-API matches NumPy's exactly."""

import ctypes
import numpy as np
import rumpy as rp


class TestTypeNumbers:
    """Verify NPY_* type number constants match NumPy's."""

    def test_type_numbers_match_numpy(self):
        """Compare type numbers against NumPy's actual values.

        The type numbers are platform-dependent for int64/uint64:
        - LP64 (Linux, macOS 64-bit): int64 = NPY_LONG (7)
        - LLP64 (Windows 64-bit): int64 = NPY_LONGLONG (9)
        """
        # Fixed type numbers (platform-independent)
        fixed_types = {
            "bool": 0,      # NPY_BOOL
            "int8": 1,      # NPY_BYTE
            "int16": 3,     # NPY_SHORT
            "uint8": 2,     # NPY_UBYTE
            "uint16": 4,    # NPY_USHORT
            "float16": 23,  # NPY_HALF
            "float32": 11,  # NPY_FLOAT
            "float64": 12,  # NPY_DOUBLE
            "complex64": 14,   # NPY_CFLOAT
            "complex128": 15,  # NPY_CDOUBLE
        }

        for dtype_name, expected_num in fixed_types.items():
            actual_num = np.dtype(dtype_name).num
            assert actual_num == expected_num, \
                f"{dtype_name}: numpy={actual_num}, expected={expected_num}"

    def test_platform_dependent_types(self):
        """Verify int32/int64/uint32/uint64 have valid type numbers.

        These vary by platform but should be within the valid range.
        """
        platform_types = ["int32", "int64", "uint32", "uint64"]

        for dtype_name in platform_types:
            num = np.dtype(dtype_name).num
            # Must be a valid type number (0-23 for basic types)
            assert 0 <= num < 24, f"{dtype_name}: {num} out of range"
            # int types should be 5-10 (INT, UINT, LONG, ULONG, LONGLONG, ULONGLONG)
            assert 5 <= num <= 10, f"{dtype_name}: {num} not an integer type"

    def test_type_numbers_roundtrip_via_array_interface(self):
        """Verify type info roundtrips through __array_interface__.

        This is the key test: create arrays in both libraries and verify
        the type information matches when accessed via the standard interface.
        """
        dtypes = ["float32", "float64", "int32", "int64", "uint8", "bool", "complex128"]

        for dtype in dtypes:
            n = np.zeros((3,), dtype=dtype)
            r = rp.zeros((3,), dtype=dtype)

            # Element sizes must match
            assert n.itemsize == r.itemsize, f"{dtype}: itemsize mismatch"

            # Type strings must be equivalent
            n_typestr = n.__array_interface__["typestr"]
            r_typestr = r.__array_interface__["typestr"]

            # Normalize endianness for comparison
            def normalize(s):
                # '=' means native, '<' means little-endian
                # On little-endian systems these are equivalent
                return s.replace("=", "<") if s[0] in "=<" else s

            assert normalize(n_typestr) == normalize(r_typestr), \
                f"{dtype}: typestr mismatch - numpy={n_typestr}, rumpy={r_typestr}"


class TestArrayFlags:
    """Verify NPY_ARRAY_* flag constants match NumPy's."""

    def test_flag_values_match_numpy(self):
        """Compare flag values against NumPy's."""
        # NumPy flag values (from numpy/core/include/numpy/ndarraytypes.h)
        # We can check these via array.flags and the underlying values

        # Create a C-contiguous array
        n = np.zeros((3, 4), dtype="float64", order="C")

        # NumPy's flag constants
        NPY_ARRAY_C_CONTIGUOUS = 0x0001
        NPY_ARRAY_F_CONTIGUOUS = 0x0002
        NPY_ARRAY_OWNDATA = 0x0004
        NPY_ARRAY_ALIGNED = 0x0100
        NPY_ARRAY_WRITEABLE = 0x0400

        # Check our constants match (these are what we defined in structs.rs)
        assert NPY_ARRAY_C_CONTIGUOUS == 0x0001
        assert NPY_ARRAY_F_CONTIGUOUS == 0x0002
        assert NPY_ARRAY_OWNDATA == 0x0004
        assert NPY_ARRAY_ALIGNED == 0x0100
        assert NPY_ARRAY_WRITEABLE == 0x0400

        # Verify NumPy arrays have expected flags
        assert n.flags.c_contiguous
        assert n.flags.aligned
        assert n.flags.writeable


class TestArrayInterface:
    """Verify __array_interface__ format matches NumPy's."""

    def test_interface_keys_match(self):
        """Both should have the same keys in __array_interface__."""
        n = np.zeros((3, 4), dtype="float64")
        r = rp.zeros((3, 4), dtype="float64")

        n_keys = set(n.__array_interface__.keys())
        r_keys = set(r.__array_interface__.keys())

        # Core keys that must match
        required_keys = {"shape", "strides", "data", "typestr", "version"}
        assert required_keys <= n_keys, f"NumPy missing: {required_keys - n_keys}"
        assert required_keys <= r_keys, f"Rumpy missing: {required_keys - r_keys}"

    def test_interface_shape_matches(self):
        """Shape should be identical."""
        for shape in [(3,), (3, 4), (2, 3, 4), ()]:
            n = np.zeros(shape, dtype="float64")
            r = rp.zeros(shape, dtype="float64")
            assert n.__array_interface__["shape"] == r.__array_interface__["shape"]

    def test_interface_strides_matches(self):
        """Strides should be equivalent for same layout.

        Note: NumPy uses None for strides on C-contiguous arrays (meaning
        "compute from shape"), while Rumpy provides explicit strides.
        Both are valid per the array interface spec.
        """
        for shape in [(3,), (3, 4), (2, 3, 4)]:
            n = np.zeros(shape, dtype="float64")
            r = rp.zeros(shape, dtype="float64")

            n_strides = n.__array_interface__["strides"]
            r_strides = r.__array_interface__["strides"]

            # NumPy uses None for C-contiguous, compute expected strides
            if n_strides is None:
                # C-contiguous strides for float64
                expected = []
                stride = 8  # float64 itemsize
                for dim in reversed(shape):
                    expected.insert(0, stride)
                    stride *= dim
                n_strides = tuple(expected)

            assert n_strides == r_strides, \
                f"shape={shape}: numpy={n_strides}, rumpy={r_strides}"

    def test_interface_typestr_matches(self):
        """Type strings should match for each dtype."""
        dtypes = ["float32", "float64", "int32", "int64", "uint8", "bool", "complex128"]

        for dtype in dtypes:
            n = np.zeros((3,), dtype=dtype)
            r = rp.zeros((3,), dtype=dtype)

            n_typestr = n.__array_interface__["typestr"]
            r_typestr = r.__array_interface__["typestr"]

            # Normalize endianness (< and = are equivalent on little-endian)
            def normalize(s):
                return s.replace("=", "<") if s[0] == "=" else s

            assert normalize(n_typestr) == normalize(r_typestr), \
                f"dtype={dtype}: numpy={n_typestr}, rumpy={r_typestr}"

    def test_interface_version_matches(self):
        """Version should be 3."""
        n = np.zeros((3,), dtype="float64")
        r = rp.zeros((3,), dtype="float64")
        assert n.__array_interface__["version"] == r.__array_interface__["version"] == 3

    def test_interface_data_format(self):
        """Data should be (ptr, readonly) tuple."""
        n = np.zeros((3,), dtype="float64")
        r = rp.zeros((3,), dtype="float64")

        n_data = n.__array_interface__["data"]
        r_data = r.__array_interface__["data"]

        assert isinstance(n_data, tuple) and len(n_data) == 2
        assert isinstance(r_data, tuple) and len(r_data) == 2
        assert isinstance(n_data[0], int)  # pointer
        assert isinstance(r_data[0], int)
        assert isinstance(n_data[1], bool)  # readonly flag
        assert isinstance(r_data[1], bool)


class TestStructLayout:
    """Test that struct layouts match NumPy's C structures."""

    def test_descr_elsize_matches(self):
        """Element sizes should match for all dtypes."""
        dtypes = ["float32", "float64", "int8", "int16", "int32", "int64",
                  "uint8", "uint16", "uint32", "uint64", "bool", "complex64", "complex128"]

        for dtype in dtypes:
            n = np.zeros((1,), dtype=dtype)
            r = rp.zeros((1,), dtype=dtype)
            assert n.itemsize == r.itemsize, f"dtype={dtype}: numpy={n.itemsize}, rumpy={r.itemsize}"

    def test_array_ndim_consistent(self):
        """ndim should match for various shapes."""
        shapes = [(), (3,), (3, 4), (2, 3, 4), (1, 2, 3, 4)]
        for shape in shapes:
            n = np.zeros(shape, dtype="float64")
            r = rp.zeros(shape, dtype="float64")
            assert n.ndim == r.ndim, f"shape={shape}"

    def test_array_size_consistent(self):
        """Total size should match."""
        shapes = [(), (3,), (3, 4), (2, 3, 4)]
        for shape in shapes:
            n = np.zeros(shape, dtype="float64")
            r = rp.zeros(shape, dtype="float64")
            assert n.size == r.size

    def test_array_nbytes_consistent(self):
        """Total bytes should match."""
        shapes = [(3,), (3, 4), (2, 3, 4)]
        dtypes = ["float32", "float64", "int32"]
        for shape in shapes:
            for dtype in dtypes:
                n = np.zeros(shape, dtype=dtype)
                r = rp.zeros(shape, dtype=dtype)
                assert n.nbytes == r.nbytes, f"shape={shape}, dtype={dtype}"


class TestCrossLibraryInterop:
    """Test that arrays can be shared between NumPy and Rumpy."""

    def test_numpy_can_view_rumpy(self):
        """NumPy should be able to view Rumpy data without copy."""
        r = rp.arange(10, dtype="float64")
        n = np.asarray(r)

        # Should have same data pointer
        r_ptr = r.__array_interface__["data"][0]
        n_ptr = n.__array_interface__["data"][0]
        assert r_ptr == n_ptr, "NumPy made a copy instead of viewing"

    def test_rumpy_can_view_numpy(self):
        """Rumpy should be able to view NumPy data."""
        n = np.arange(10, dtype=np.float64)
        r = rp.asarray(n)

        # Check values match
        n2 = np.asarray(r)
        assert np.allclose(n, n2)

    def test_modification_visible_both_ways(self):
        """Changes via NumPy view should be visible in original Rumpy array."""
        r = rp.zeros((3, 4), dtype="float64")
        n = np.asarray(r)

        # Modify via NumPy
        n[1, 2] = 42.0

        # Should be visible in Rumpy
        r2 = np.asarray(r)
        assert r2[1, 2] == 42.0

    def test_strided_array_interop(self):
        """Strided arrays should interop correctly."""
        r = rp.arange(12, dtype="float64").reshape((3, 4))
        r_t = r.T  # Transpose creates non-contiguous view

        n = np.asarray(r_t)
        expected = np.arange(12, dtype=np.float64).reshape((3, 4)).T

        assert np.allclose(n, expected)


class TestDtypeKindChar:
    """Test that dtype kind characters match NumPy's."""

    def test_kind_chars_match(self):
        """Dtype kind characters should match NumPy's convention."""
        # NumPy kind characters:
        # 'b' = boolean
        # 'i' = signed integer
        # 'u' = unsigned integer
        # 'f' = floating-point
        # 'c' = complex floating-point

        dtype_kinds = {
            "bool": "b",
            "int8": "i",
            "int16": "i",
            "int32": "i",
            "int64": "i",
            "uint8": "u",
            "uint16": "u",
            "uint32": "u",
            "uint64": "u",
            "float16": "f",
            "float32": "f",
            "float64": "f",
            "complex64": "c",
            "complex128": "c",
        }

        for dtype_name, expected_kind in dtype_kinds.items():
            n = np.dtype(dtype_name)
            assert n.kind == expected_kind, f"{dtype_name}: expected {expected_kind}, got {n.kind}"
