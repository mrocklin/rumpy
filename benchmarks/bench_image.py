"""Image processing benchmarks.

Each benchmark is a function that takes (xp, size) where xp is numpy or rumpy,
and returns a callable that performs the benchmark.
"""

import numpy as np


def grayscale_convert(xp, size):
    """Convert RGB to grayscale: 0.299*R + 0.587*G + 0.114*B"""
    n = int(np.sqrt(size / 3))
    rgb = xp.array(np.random.rand(n, n, 3))
    weights = xp.array([0.299, 0.587, 0.114])

    def fn():
        return (rgb * weights).sum(axis=2)
    return fn


def image_normalize(xp, size):
    """Normalize image to [0, 1] range."""
    n = int(np.sqrt(size))
    # Simulate uint8-like values
    img = xp.array(np.random.randint(0, 256, (n, n)).astype(np.float64))

    def fn():
        return (img - img.min()) / (img.max() - img.min())
    return fn


def sobel_edges(xp, size):
    """Sobel edge detection (gradient magnitude)."""
    n = int(np.sqrt(size))
    img = xp.array(np.random.rand(n, n))

    def fn():
        # Horizontal gradient: diff along axis 1
        gx = img[:, 2:] - img[:, :-2]
        # Vertical gradient: diff along axis 0
        gy = img[2:, :] - img[:-2, :]
        # Crop to common shape and compute magnitude
        gx_crop = gx[1:-1, :]
        gy_crop = gy[:, 1:-1]
        return xp.sqrt(gx_crop**2 + gy_crop**2)
    return fn


def histogram_equalization_prep(xp, size):
    """Compute histogram (bincount) for histogram equalization."""
    n = int(np.sqrt(size))
    # Simulate uint8 values
    img = xp.array(np.random.randint(0, 256, (n, n)))

    def fn():
        flat = img.flatten()
        hist = xp.bincount(flat, minlength=256)
        return xp.cumsum(hist)
    return fn


def box_blur_separable(xp, size):
    """Separable box blur: blur rows then columns."""
    n = int(np.sqrt(size))
    img = xp.array(np.random.rand(n, n))
    k = 3  # kernel size

    def fn():
        # Blur along axis 1 (horizontal)
        blurred_h = (img[:, :-2] + img[:, 1:-1] + img[:, 2:]) / k
        # Blur along axis 0 (vertical)
        blurred = (blurred_h[:-2, :] + blurred_h[1:-1, :] + blurred_h[2:, :]) / k
        return blurred
    return fn


def threshold_binary(xp, size):
    """Binary thresholding."""
    n = int(np.sqrt(size))
    img = xp.array(np.random.rand(n, n))

    def fn():
        threshold = 0.5
        return (img > threshold).astype('float64')
    return fn


def image_resize_nearest(xp, size):
    """Nearest-neighbor resize (2x downscale)."""
    n = int(np.sqrt(size))
    # Make sure n is even
    n = n - (n % 2)
    img = xp.array(np.random.rand(n, n))

    def fn():
        # Downsample by taking every other pixel
        return img[::2, ::2]
    return fn


def image_flip(xp, size):
    """Flip image horizontally and vertically."""
    n = int(np.sqrt(size))
    img = xp.array(np.random.rand(n, n))

    def fn():
        flipped_h = img[:, ::-1]
        flipped_v = img[::-1, :]
        return flipped_h + flipped_v
    return fn


def contrast_stretch(xp, size):
    """Contrast stretching: clip and rescale."""
    n = int(np.sqrt(size))
    img = xp.array(np.random.rand(n, n))

    def fn():
        low, high = 0.2, 0.8
        clipped = xp.clip(img, low, high)
        return (clipped - low) / (high - low)
    return fn


def channel_swap(xp, size):
    """Swap RGB channels to BGR."""
    n = int(np.sqrt(size / 3))
    rgb = xp.array(np.random.rand(n, n, 3))

    def fn():
        # Requires advanced indexing or flip on last axis
        return rgb[:, :, ::-1]
    return fn


BENCHMARKS = [
    ("grayscale convert", grayscale_convert),
    ("image normalize", image_normalize),
    ("Sobel edges", sobel_edges),
    ("histogram equalization prep", histogram_equalization_prep),  # requires bincount
    ("box blur (separable)", box_blur_separable),
    ("binary threshold", threshold_binary),  # requires astype
    ("nearest resize (2x down)", image_resize_nearest),
    ("image flip H+V", image_flip),  # requires negative strides
    ("contrast stretch", contrast_stretch),  # requires clip
    ("channel swap (RGB->BGR)", channel_swap),  # requires negative stride on axis
]
