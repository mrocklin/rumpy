"""Bitwise operation benchmarks.

Each benchmark is a function that takes (xp, size) where xp is numpy or rumpy,
and returns a callable that performs the benchmark.
"""

import numpy as np


def mask_and(xp, size):
    """Bitwise AND for masking."""
    a = xp.array(np.random.randint(0, 256, size, dtype='int64'))
    mask = xp.array(np.full(size, 0x0F, dtype='int64'))  # lower 4 bits

    def fn():
        return a & mask
    return fn


def mask_or(xp, size):
    """Bitwise OR for setting flags."""
    a = xp.array(np.random.randint(0, 256, size, dtype='int64'))
    flags = xp.array(np.full(size, 0x80, dtype='int64'))  # set high bit

    def fn():
        return a | flags
    return fn


def xor_toggle(xp, size):
    """Bitwise XOR for toggling bits."""
    a = xp.array(np.random.randint(0, 256, size, dtype='int64'))
    toggle = xp.array(np.full(size, 0xFF, dtype='int64'))

    def fn():
        return a ^ toggle
    return fn


def invert_bits(xp, size):
    """Bitwise NOT / invert."""
    a = xp.array(np.random.randint(0, 256, size, dtype='int64'))

    def fn():
        return xp.invert(a)
    return fn


def left_shift_mul(xp, size):
    """Left shift (multiply by power of 2)."""
    a = xp.array(np.random.randint(0, 1000, size, dtype='int64'))

    def fn():
        return a << 3  # multiply by 8
    return fn


def right_shift_div(xp, size):
    """Right shift (divide by power of 2)."""
    a = xp.array(np.random.randint(0, 10000, size, dtype='int64'))

    def fn():
        return a >> 2  # divide by 4
    return fn


def extract_bits(xp, size):
    """Extract specific bits (common in packed data)."""
    packed = xp.array(np.random.randint(0, 65536, size, dtype='int64'))

    def fn():
        # Extract bits 4-7 (second nibble)
        return (packed >> 4) & 0x0F
    return fn


def pack_bits(xp, size):
    """Pack two values into one (interleaving)."""
    lo = xp.array(np.random.randint(0, 16, size, dtype='int64'))
    hi = xp.array(np.random.randint(0, 16, size, dtype='int64'))

    def fn():
        return (hi << 4) | lo
    return fn


def flag_check(xp, size):
    """Check multiple flags with AND."""
    flags = xp.array(np.random.randint(0, 256, size, dtype='int64'))
    required = 0x05  # bits 0 and 2 must be set

    def fn():
        return (flags & required) == required
    return fn


def bit_count_manual(xp, size):
    """Manual popcount via Brian Kernighan's algorithm (simplified)."""
    # This tests chained bitwise ops
    a = xp.array(np.random.randint(0, 256, size, dtype='int64'))

    def fn():
        # Count bits set in each byte using parallel sum
        # This is a simplified version - real popcount uses lookup or intrinsics
        b = a - ((a >> 1) & 0x55)
        c = (b & 0x33) + ((b >> 2) & 0x33)
        return (c + (c >> 4)) & 0x0F
    return fn


def rgb_extract(xp, size):
    """Extract RGB channels from packed 24-bit color."""
    # Simulate packed RGB values (0xRRGGBB)
    packed = xp.array(np.random.randint(0, 0xFFFFFF, size, dtype='int64'))

    def fn():
        r = (packed >> 16) & 0xFF
        g = (packed >> 8) & 0xFF
        b = packed & 0xFF
        return r + g + b  # combine to avoid returning tuple
    return fn


def gray_code(xp, size):
    """Convert binary to Gray code."""
    a = xp.array(np.random.randint(0, 1000, size, dtype='int64'))

    def fn():
        return a ^ (a >> 1)
    return fn


BENCHMARKS = [
    ("mask AND", mask_and),
    ("mask OR", mask_or),
    ("XOR toggle", xor_toggle),
    ("invert", invert_bits),
    ("left shift", left_shift_mul),
    ("right shift", right_shift_div),
    ("extract bits", extract_bits),
    ("pack bits", pack_bits),
    ("flag check", flag_check),
    ("bit count (parallel)", bit_count_manual),
    ("RGB extract", rgb_extract),
    ("Gray code", gray_code),
]
