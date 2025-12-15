"""Test FFT module against numpy.fft."""

import numpy as np
import pytest
import rumpy as rp
from helpers import assert_eq

# Seed for reproducibility
np.random.seed(42)

# Test dtypes covering different sizes and types
FFT_DTYPES = [
    np.float32,
    np.float64,
    np.int32,
    np.int64,
    np.complex64,
    np.complex128,
]

# 1D shapes: power-of-2, non-power-of-2, small, larger
SHAPES_1D = [
    (4,),
    (32,),
    (64,),
    (37,),  # prime
    (100,),
]

# 2D shapes: square, non-square, small, larger
SHAPES_2D = [
    (4, 4),
    (8, 8),
    (8, 16),
    (16, 8),
    (7, 11),  # primes
]


class TestFftParametrized:
    """Parametrized tests across dtypes and shapes."""

    @pytest.mark.parametrize("dtype", FFT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_1D)
    def test_fft_dtypes_shapes(self, dtype, shape):
        """FFT works across dtypes and 1D shapes."""
        if np.issubdtype(dtype, np.complexfloating):
            data = (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(dtype)
        else:
            data = np.random.randn(*shape).astype(dtype)
        r = rp.fft.fft(rp.asarray(data))
        n = np.fft.fft(data)
        assert_eq(r, n, rtol=1e-5)

    @pytest.mark.parametrize("dtype", FFT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_1D)
    def test_ifft_dtypes_shapes(self, dtype, shape):
        """IFFT works across dtypes and 1D shapes."""
        if np.issubdtype(dtype, np.complexfloating):
            data = (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(dtype)
        else:
            data = np.random.randn(*shape).astype(dtype)
        r = rp.fft.ifft(rp.asarray(data))
        n = np.fft.ifft(data)
        assert_eq(r, n, rtol=1e-5)

    @pytest.mark.parametrize("dtype", FFT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_2D)
    def test_fft2_dtypes_shapes(self, dtype, shape):
        """FFT2 works across dtypes and 2D shapes."""
        if np.issubdtype(dtype, np.complexfloating):
            data = (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(dtype)
        else:
            data = np.random.randn(*shape).astype(dtype)
        r = rp.fft.fft2(rp.asarray(data))
        n = np.fft.fft2(data)
        assert_eq(r, n, rtol=1e-5)

    @pytest.mark.parametrize("dtype", FFT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_2D)
    def test_ifft2_dtypes_shapes(self, dtype, shape):
        """IFFT2 works across dtypes and 2D shapes."""
        if np.issubdtype(dtype, np.complexfloating):
            data = (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(dtype)
        else:
            data = np.random.randn(*shape).astype(dtype)
        r = rp.fft.ifft2(rp.asarray(data))
        n = np.fft.ifft2(data)
        assert_eq(r, n, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
    @pytest.mark.parametrize("shape", SHAPES_1D)
    def test_rfft_dtypes_shapes(self, dtype, shape):
        """RFFT works across real dtypes and 1D shapes."""
        data = np.random.randn(*shape).astype(dtype)
        r = rp.fft.rfft(rp.asarray(data))
        n = np.fft.rfft(data)
        assert_eq(r, n, rtol=1e-5)

    @pytest.mark.parametrize("shape", SHAPES_1D)
    def test_irfft_shapes(self, shape):
        """IRFFT works across 1D shapes."""
        data = np.random.randn(*shape)
        freq = np.fft.rfft(data)
        r = rp.fft.irfft(rp.asarray(freq))
        n = np.fft.irfft(freq)
        assert_eq(r, n, rtol=1e-5)

    @pytest.mark.parametrize("dtype", FFT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_1D)
    def test_fftshift_1d_dtypes_shapes(self, dtype, shape):
        """FFTSHIFT works across dtypes and 1D shapes."""
        if np.issubdtype(dtype, np.complexfloating):
            data = (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(dtype)
        else:
            data = np.random.randn(*shape).astype(dtype)
        freq = np.fft.fft(data).astype(np.complex128)  # Ensure complex128
        r = rp.fft.fftshift(rp.asarray(freq))
        n = np.fft.fftshift(freq)
        assert_eq(r, n, rtol=1e-5)

    @pytest.mark.parametrize("dtype", FFT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_2D)
    def test_fftshift_2d_dtypes_shapes(self, dtype, shape):
        """FFTSHIFT works across dtypes and 2D shapes."""
        if np.issubdtype(dtype, np.complexfloating):
            data = (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(dtype)
        else:
            data = np.random.randn(*shape).astype(dtype)
        freq = np.fft.fft2(data).astype(np.complex128)  # Ensure complex128
        r = rp.fft.fftshift(rp.asarray(freq))
        n = np.fft.fftshift(freq)
        assert_eq(r, n, rtol=1e-5)

    @pytest.mark.parametrize("dtype", FFT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_1D)
    def test_ifftshift_1d_dtypes_shapes(self, dtype, shape):
        """IFFTSHIFT works across dtypes and 1D shapes."""
        if np.issubdtype(dtype, np.complexfloating):
            data = (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(dtype)
        else:
            data = np.random.randn(*shape).astype(dtype)
        freq = np.fft.fftshift(np.fft.fft(data)).astype(np.complex128)  # Ensure complex128
        r = rp.fft.ifftshift(rp.asarray(freq))
        n = np.fft.ifftshift(freq)
        assert_eq(r, n, rtol=1e-5)

    @pytest.mark.parametrize("n", [8, 16, 32, 37, 64, 100])
    @pytest.mark.parametrize("d", [1.0, 0.1, 0.5])
    def test_fftfreq_params(self, n, d):
        """FFTFREQ works across n and d values."""
        r = rp.fft.fftfreq(n, d=d)
        n_result = np.fft.fftfreq(n, d=d)
        assert_eq(r, n_result)

    @pytest.mark.parametrize("n", [8, 16, 32, 37, 64, 100])
    @pytest.mark.parametrize("d", [1.0, 0.1, 0.5])
    def test_rfftfreq_params(self, n, d):
        """RFFTFREQ works across n and d values."""
        r = rp.fft.rfftfreq(n, d=d)
        n_result = np.fft.rfftfreq(n, d=d)
        assert_eq(r, n_result)


class TestFftRoundtrip:
    """Test FFT roundtrip consistency across dtypes."""

    @pytest.mark.parametrize("dtype", FFT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_1D)
    def test_fft_ifft_roundtrip(self, dtype, shape):
        """FFT then IFFT returns original."""
        if np.issubdtype(dtype, np.complexfloating):
            data = (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(dtype)
        else:
            data = np.random.randn(*shape).astype(dtype)
        rp_arr = rp.asarray(data)
        result = rp.fft.ifft(rp.fft.fft(rp_arr))
        assert_eq(result, data.astype(np.complex128), rtol=1e-10)

    @pytest.mark.parametrize("dtype", FFT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_2D)
    def test_fft2_ifft2_roundtrip(self, dtype, shape):
        """FFT2 then IFFT2 returns original."""
        if np.issubdtype(dtype, np.complexfloating):
            data = (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(dtype)
        else:
            data = np.random.randn(*shape).astype(dtype)
        rp_arr = rp.asarray(data)
        result = rp.fft.ifft2(rp.fft.fft2(rp_arr))
        assert_eq(result, data.astype(np.complex128), rtol=1e-10)

    @pytest.mark.parametrize("shape", [(4,), (32,), (64,), (100,)])  # Even lengths only
    def test_rfft_irfft_roundtrip(self, shape):
        """RFFT then IRFFT returns original (even length only)."""
        # Note: irfft defaults to even length output; odd-length roundtrip requires n param
        data = np.random.randn(*shape)
        rp_arr = rp.asarray(data)
        result = rp.fft.irfft(rp.fft.rfft(rp_arr))
        assert_eq(result, data, rtol=1e-10)

    @pytest.mark.parametrize("shape", SHAPES_1D + SHAPES_2D)
    def test_fftshift_ifftshift_roundtrip(self, shape):
        """FFTSHIFT then IFFTSHIFT returns original."""
        data = np.random.randn(*shape) + 1j * np.random.randn(*shape)
        rp_arr = rp.asarray(data)
        result = rp.fft.ifftshift(rp.fft.fftshift(rp_arr))
        assert_eq(result, data, rtol=1e-10)


class TestFft:
    """Test basic fft/ifft functions."""

    def test_fft_1d_simple(self):
        """FFT of simple 1D array."""
        data = [1.0, 2.0, 3.0, 4.0]
        r = rp.fft.fft(rp.asarray(data))
        n = np.fft.fft(np.array(data))
        assert_eq(r, n)

    def test_fft_1d_powers_of_2(self):
        """FFT works well with power-of-2 lengths."""
        data = np.random.randn(64)
        r = rp.fft.fft(rp.asarray(data))
        n = np.fft.fft(data)
        assert_eq(r, n)

    def test_fft_1d_non_power_of_2(self):
        """FFT handles non-power-of-2 lengths."""
        data = np.random.randn(37)
        r = rp.fft.fft(rp.asarray(data))
        n = np.fft.fft(data)
        assert_eq(r, n)

    def test_ifft_1d(self):
        """Inverse FFT recovers original signal."""
        data = np.random.randn(32)
        r = rp.fft.ifft(rp.fft.fft(rp.asarray(data)))
        n = np.fft.ifft(np.fft.fft(data))
        assert_eq(r, n)

    def test_fft_ifft_roundtrip(self):
        """FFT then IFFT returns original."""
        data = np.random.randn(64)
        rp_arr = rp.asarray(data)
        result = rp.fft.ifft(rp.fft.fft(rp_arr))
        assert_eq(result, data, rtol=1e-10)

    def test_fft_complex_input(self):
        """FFT of complex input."""
        real = np.random.randn(32)
        imag = np.random.randn(32)
        data = real + 1j * imag
        r = rp.fft.fft(rp.asarray(data))
        n = np.fft.fft(data)
        assert_eq(r, n)


class TestFft2:
    """Test 2D FFT functions."""

    def test_fft2_simple(self):
        """2D FFT of simple array."""
        data = np.random.randn(8, 8)
        r = rp.fft.fft2(rp.asarray(data))
        n = np.fft.fft2(data)
        assert_eq(r, n)

    def test_ifft2_simple(self):
        """2D inverse FFT."""
        data = np.random.randn(8, 8)
        r = rp.fft.ifft2(rp.fft.fft2(rp.asarray(data)))
        n = np.fft.ifft2(np.fft.fft2(data))
        assert_eq(r, n)

    def test_fft2_non_square(self):
        """2D FFT of non-square array."""
        data = np.random.randn(16, 8)
        r = rp.fft.fft2(rp.asarray(data))
        n = np.fft.fft2(data)
        assert_eq(r, n)


class TestRfft:
    """Test real FFT functions (optimized for real input)."""

    def test_rfft_1d(self):
        """Real FFT of 1D array."""
        data = np.random.randn(64)
        r = rp.fft.rfft(rp.asarray(data))
        n = np.fft.rfft(data)
        assert_eq(r, n)

    def test_irfft_1d(self):
        """Inverse real FFT."""
        data = np.random.randn(64)
        freq = np.fft.rfft(data)
        r = rp.fft.irfft(rp.asarray(freq))
        n = np.fft.irfft(freq)
        assert_eq(r, n)

    def test_rfft_irfft_roundtrip(self):
        """rfft then irfft returns original."""
        data = np.random.randn(64)
        rp_arr = rp.asarray(data)
        result = rp.fft.irfft(rp.fft.rfft(rp_arr))
        assert_eq(result, data, rtol=1e-10)


class TestFftshift:
    """Test fftshift/ifftshift functions."""

    def test_fftshift_1d(self):
        """Shift zero-frequency to center."""
        data = np.fft.fft(np.random.randn(32))
        r = rp.fft.fftshift(rp.asarray(data))
        n = np.fft.fftshift(data)
        assert_eq(r, n)

    def test_ifftshift_1d(self):
        """Inverse of fftshift."""
        data = np.fft.fftshift(np.fft.fft(np.random.randn(32)))
        r = rp.fft.ifftshift(rp.asarray(data))
        n = np.fft.ifftshift(data)
        assert_eq(r, n)

    def test_fftshift_2d(self):
        """2D fftshift."""
        data = np.fft.fft2(np.random.randn(8, 8))
        r = rp.fft.fftshift(rp.asarray(data))
        n = np.fft.fftshift(data)
        assert_eq(r, n)


class TestFftfreq:
    """Test frequency generation functions."""

    def test_fftfreq(self):
        """Generate FFT sample frequencies."""
        r = rp.fft.fftfreq(8, d=1.0)
        n = np.fft.fftfreq(8, d=1.0)
        assert_eq(r, n)

    def test_fftfreq_custom_spacing(self):
        """fftfreq with custom sample spacing."""
        r = rp.fft.fftfreq(16, d=0.1)
        n = np.fft.fftfreq(16, d=0.1)
        assert_eq(r, n)

    def test_rfftfreq(self):
        """Generate real FFT sample frequencies."""
        r = rp.fft.rfftfreq(8, d=1.0)
        n = np.fft.rfftfreq(8, d=1.0)
        assert_eq(r, n)


class TestFftn:
    """Test n-dimensional FFT functions."""

    @pytest.mark.parametrize("shape", [(8,), (4, 4), (2, 3, 4), (2, 2, 2, 2)])
    def test_fftn_shapes(self, shape):
        """fftn works across different dimensions."""
        data = np.random.randn(*shape)
        r = rp.fft.fftn(rp.asarray(data))
        n = np.fft.fftn(data)
        assert_eq(r, n, rtol=1e-10)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_fftn_dtypes(self, dtype):
        """fftn works across different dtypes."""
        if np.issubdtype(dtype, np.complexfloating):
            data = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(dtype)
        else:
            data = np.random.randn(4, 4).astype(dtype)
        r = rp.fft.fftn(rp.asarray(data))
        n = np.fft.fftn(data)
        assert_eq(r, n, rtol=1e-5)

    def test_fftn_axes_subset(self):
        """fftn with axes parameter."""
        data = np.random.randn(4, 6, 8)
        # Transform only last two axes
        r = rp.fft.fftn(rp.asarray(data), axes=[1, 2])
        n = np.fft.fftn(data, axes=[1, 2])
        assert_eq(r, n, rtol=1e-10)

    def test_fftn_single_axis(self):
        """fftn with single axis is like fft along that axis."""
        data = np.random.randn(4, 6)
        r = rp.fft.fftn(rp.asarray(data), axes=[0])
        n = np.fft.fftn(data, axes=[0])
        assert_eq(r, n, rtol=1e-10)


class TestIfftn:
    """Test n-dimensional inverse FFT functions."""

    @pytest.mark.parametrize("shape", [(8,), (4, 4), (2, 3, 4), (2, 2, 2, 2)])
    def test_ifftn_shapes(self, shape):
        """ifftn works across different dimensions."""
        data = np.random.randn(*shape) + 1j * np.random.randn(*shape)
        r = rp.fft.ifftn(rp.asarray(data))
        n = np.fft.ifftn(data)
        assert_eq(r, n, rtol=1e-10)

    def test_ifftn_axes_subset(self):
        """ifftn with axes parameter."""
        data = np.random.randn(4, 6, 8) + 1j * np.random.randn(4, 6, 8)
        r = rp.fft.ifftn(rp.asarray(data), axes=[1, 2])
        n = np.fft.ifftn(data, axes=[1, 2])
        assert_eq(r, n, rtol=1e-10)

    @pytest.mark.parametrize("shape", [(8,), (4, 4), (2, 3, 4)])
    def test_fftn_ifftn_roundtrip(self, shape):
        """fftn then ifftn returns original."""
        data = np.random.randn(*shape)
        rp_arr = rp.asarray(data)
        result = rp.fft.ifftn(rp.fft.fftn(rp_arr))
        assert_eq(result, data, rtol=1e-10)


class TestRfftn:
    """Test n-dimensional real FFT functions."""

    @pytest.mark.parametrize("shape", [(8,), (4, 6), (2, 4, 6)])
    def test_rfftn_shapes(self, shape):
        """rfftn works across different dimensions."""
        data = np.random.randn(*shape)
        r = rp.fft.rfftn(rp.asarray(data))
        n = np.fft.rfftn(data)
        assert_eq(r, n, rtol=1e-10)

    def test_rfftn_output_shape(self):
        """rfftn output has correct shape (last axis truncated)."""
        data = np.random.randn(4, 8)
        r = rp.fft.rfftn(rp.asarray(data))
        n = np.fft.rfftn(data)
        assert r.shape == n.shape
        assert r.shape == (4, 5)  # 8 // 2 + 1 = 5

    def test_rfftn_axes_subset(self):
        """rfftn with axes parameter."""
        data = np.random.randn(4, 6, 8)
        r = rp.fft.rfftn(rp.asarray(data), axes=[1, 2])
        n = np.fft.rfftn(data, axes=[1, 2])
        assert_eq(r, n, rtol=1e-10)


class TestIrfftn:
    """Test n-dimensional inverse real FFT functions."""

    @pytest.mark.parametrize("shape", [(8,), (4, 6), (2, 4, 6)])
    def test_irfftn_shapes(self, shape):
        """irfftn works across different dimensions (even output)."""
        data = np.random.randn(*shape)
        freq = np.fft.rfftn(data)
        r = rp.fft.irfftn(rp.asarray(freq))
        n = np.fft.irfftn(freq)
        assert_eq(r, n, rtol=1e-10)

    @pytest.mark.parametrize("shape", [(8,), (4, 8), (2, 4, 8)])
    def test_rfftn_irfftn_roundtrip(self, shape):
        """rfftn then irfftn returns original (even length last axis)."""
        data = np.random.randn(*shape)
        rp_arr = rp.asarray(data)
        result = rp.fft.irfftn(rp.fft.rfftn(rp_arr))
        assert_eq(result, data, rtol=1e-10)


class TestRfft2:
    """Test 2D real FFT functions."""

    @pytest.mark.parametrize("shape", [(4, 4), (8, 16), (6, 10)])
    def test_rfft2_shapes(self, shape):
        """rfft2 works across different 2D shapes."""
        data = np.random.randn(*shape)
        r = rp.fft.rfft2(rp.asarray(data))
        n = np.fft.rfft2(data)
        assert_eq(r, n, rtol=1e-10)

    def test_rfft2_output_shape(self):
        """rfft2 output has correct shape."""
        data = np.random.randn(4, 8)
        r = rp.fft.rfft2(rp.asarray(data))
        n = np.fft.rfft2(data)
        assert r.shape == n.shape
        assert r.shape == (4, 5)


class TestIrfft2:
    """Test 2D inverse real FFT functions."""

    @pytest.mark.parametrize("shape", [(4, 4), (8, 16), (6, 10)])
    def test_irfft2_shapes(self, shape):
        """irfft2 works across different 2D shapes (even output)."""
        data = np.random.randn(*shape)
        freq = np.fft.rfft2(data)
        r = rp.fft.irfft2(rp.asarray(freq))
        n = np.fft.irfft2(freq)
        assert_eq(r, n, rtol=1e-10)

    @pytest.mark.parametrize("shape", [(4, 8), (8, 16), (6, 10)])
    def test_rfft2_irfft2_roundtrip(self, shape):
        """rfft2 then irfft2 returns original (even last axis)."""
        data = np.random.randn(*shape)
        rp_arr = rp.asarray(data)
        result = rp.fft.irfft2(rp.fft.rfft2(rp_arr))
        assert_eq(result, data, rtol=1e-10)


class TestHfft:
    """Test Hermitian FFT functions."""

    @pytest.mark.parametrize("n", [8, 16, 32, 64])
    def test_hfft_basic(self, n):
        """hfft of Hermitian signal."""
        # Create a real signal and get its FFT
        real_signal = np.random.randn(n)
        freq = np.fft.rfft(real_signal)
        # hfft should recover the signal (up to scale/conjugation)
        r = rp.fft.hfft(rp.asarray(freq))
        n_result = np.fft.hfft(freq)
        assert_eq(r, n_result, rtol=1e-10)

    def test_hfft_with_n_param(self):
        """hfft with explicit output length."""
        data = np.array([1+0j, 2+1j, 3+0j, 2-1j])[:3]  # Hermitian symmetric (partial)
        r = rp.fft.hfft(rp.asarray(data), n=8)
        n_result = np.fft.hfft(data, n=8)
        assert_eq(r, n_result, rtol=1e-10)


class TestIhfft:
    """Test inverse Hermitian FFT functions."""

    @pytest.mark.parametrize("n", [8, 16, 32, 64])
    def test_ihfft_basic(self, n):
        """ihfft of real signal."""
        real_signal = np.random.randn(n)
        r = rp.fft.ihfft(rp.asarray(real_signal))
        n_result = np.fft.ihfft(real_signal)
        assert_eq(r, n_result, rtol=1e-10)

    def test_ihfft_output_shape(self):
        """ihfft output has correct shape."""
        data = np.random.randn(8)
        r = rp.fft.ihfft(rp.asarray(data))
        n_result = np.fft.ihfft(data)
        assert r.shape == n_result.shape
        assert r.shape == (5,)  # 8 // 2 + 1

    @pytest.mark.parametrize("n", [8, 16, 32])
    def test_hfft_ihfft_roundtrip(self, n):
        """hfft then ihfft returns original (up to truncation)."""
        freq = np.random.randn(n // 2 + 1) + 1j * np.random.randn(n // 2 + 1)
        # Make it Hermitian-ish
        freq = freq.real.astype(complex)
        np_hfft = np.fft.hfft(freq)
        np_ihfft = np.fft.ihfft(np_hfft)
        r_hfft = rp.fft.hfft(rp.asarray(freq))
        r_ihfft = rp.fft.ihfft(r_hfft)
        assert_eq(r_ihfft, np_ihfft, rtol=1e-10)
