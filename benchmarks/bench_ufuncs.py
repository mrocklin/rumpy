"""Chained ufunc benchmarks - realistic element-wise workloads.

Each benchmark is a function that takes (xp, size) where xp is numpy or rumpy,
and returns a callable that performs the benchmark.
"""

import numpy as np


def softmax(xp, size):
    """Softmax: exp(x - max) / sum(exp(x - max))"""
    data = np.random.randn(size)
    x = xp.array(data)

    def fn():
        x_max = x.max()
        e = xp.exp(x - x_max)
        return e / e.sum()
    return fn


def sigmoid(xp, size):
    """Sigmoid: 1 / (1 + exp(-x))"""
    data = np.random.randn(size)
    x = xp.array(data)

    def fn():
        return 1.0 / (1.0 + xp.exp(-x))
    return fn


def normalize(xp, size):
    """Z-score normalization: (x - mean) / std"""
    data = np.random.randn(size)
    x = xp.array(data)

    def fn():
        return (x - x.mean()) / x.std()
    return fn


def relu_scale_bias(xp, size):
    """ReLU + scale + bias: max(0, x) * scale + bias"""
    data = np.random.randn(size)
    x = xp.array(data)
    zero = xp.zeros(1)
    scale, bias = 0.1, 0.5

    def fn():
        return xp.maximum(x, zero) * scale + bias
    return fn


def log_softmax(xp, size):
    """Log-softmax: x - max - log(sum(exp(x - max)))"""
    data = np.random.randn(size)
    x = xp.array(data)

    def fn():
        x_max = x.max()
        shifted = x - x_max
        # xp.log needs to accept scalar - this tests that
        return shifted - xp.log(xp.exp(shifted).sum())
    return fn


def gelu(xp, size):
    """GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""
    data = np.random.randn(size)
    x = xp.array(data)
    sqrt_2_pi = np.sqrt(2.0 / np.pi)

    def fn():
        # Tests scalar * array (0.5 * x) - requires __rmul__ or __array_ufunc__
        inner = sqrt_2_pi * (x + 0.044715 * x**3)
        return 0.5 * x * (1.0 + xp.tanh(inner))
    return fn


def swish(xp, size):
    """Swish activation: x * sigmoid(x)"""
    data = np.random.randn(size)
    x = xp.array(data)

    def fn():
        return x * (1.0 / (1.0 + xp.exp(-x)))
    return fn


def layer_norm(xp, size):
    """Layer normalization: (x - mean) / sqrt(var + eps) * gamma + beta"""
    data = np.random.randn(size)
    x = xp.array(data)
    gamma = xp.ones(size)
    beta = xp.zeros(size)
    eps = 1e-5

    def fn():
        mean = x.mean()
        var = x.var()
        return (x - mean) / xp.sqrt(var + eps) * gamma + beta
    return fn


BENCHMARKS = [
    ("softmax", softmax),
    ("sigmoid", sigmoid),
    ("normalize (z-score)", normalize),
    ("relu + scale + bias", relu_scale_bias),
    ("log-softmax", log_softmax),  # requires xp.log(scalar)
    ("GELU activation", gelu),  # requires scalar * array
    ("swish", swish),
    ("layer norm", layer_norm),
]
