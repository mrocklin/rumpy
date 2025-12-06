"""Signal processing benchmarks.

Each benchmark is a function that takes (xp, size) where xp is numpy or rumpy,
and returns a callable that performs the benchmark.
"""

import numpy as np


def fft_roundtrip(xp, size):
    """FFT forward then inverse (roundtrip)."""
    x = xp.array(np.random.randn(size))

    def fn():
        return xp.fft.ifft(xp.fft.fft(x))
    return fn


def fft2_roundtrip(xp, size):
    """2D FFT roundtrip."""
    n = int(np.sqrt(size))
    x = xp.array(np.random.randn(n, n))

    def fn():
        return xp.fft.ifft2(xp.fft.fft2(x))
    return fn


def power_spectrum(xp, size):
    """Compute power spectrum: |FFT(x)|^2."""
    x = xp.array(np.random.randn(size))

    def fn():
        f = xp.fft.fft(x)
        return xp.real(f * xp.conj(f))
    return fn


def fft_lowpass_filter(xp, size):
    """FFT-based low-pass filter: FFT -> zero high freqs -> IFFT."""
    x = xp.array(np.random.randn(size))
    cutoff = size // 4

    def fn():
        f = xp.fft.fft(x)
        # Zero out high frequencies - requires slice assignment
        f[cutoff:-cutoff] = 0
        return xp.fft.ifft(f)
    return fn


def gradient_1d(xp, size):
    """First-order gradient via diff."""
    data = np.cumsum(np.random.randn(size))  # smooth signal
    x = xp.array(data)

    def fn():
        return xp.diff(x)
    return fn


def second_derivative(xp, size):
    """Second derivative via diff(diff(x))."""
    data = np.cumsum(np.cumsum(np.random.randn(size)))
    x = xp.array(data)

    def fn():
        return xp.diff(xp.diff(x))
    return fn


def convolve_1d(xp, size):
    """1D convolution with a small kernel."""
    x = xp.array(np.random.randn(size))
    kernel = xp.array([0.25, 0.5, 0.25])  # smoothing kernel

    def fn():
        return xp.convolve(x, kernel, mode='same')
    return fn


BENCHMARKS = [
    ("FFT roundtrip", fft_roundtrip),
    ("FFT2 roundtrip", fft2_roundtrip),
    ("power spectrum", power_spectrum),
    ("FFT lowpass filter", fft_lowpass_filter),  # requires slice assignment
    ("gradient (diff)", gradient_1d),
    ("second derivative", second_derivative),
    ("convolve 1D", convolve_1d),  # requires xp.convolve
]
