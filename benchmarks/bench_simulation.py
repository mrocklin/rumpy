"""Simulation and scientific computing benchmarks.

Each benchmark is a function that takes (xp, size) where xp is numpy or rumpy,
and returns a callable that performs the benchmark.
"""

import numpy as np


def monte_carlo_pi(xp, size):
    """Monte Carlo estimation of pi: count points inside unit circle."""
    rng = xp.random.default_rng(42)

    def fn():
        x = rng.uniform(-1, 1, size)
        y = rng.uniform(-1, 1, size)
        inside = (x*x + y*y) < 1.0
        return 4.0 * inside.sum() / size
    return fn


def pairwise_distances(xp, size):
    """Compute pairwise squared distances between N points in 2D."""
    n = int(np.sqrt(size))
    points = xp.array(np.random.randn(n, 2))

    def fn():
        # Broadcasting: (n,1,2) - (1,n,2) -> (n,n,2) -> sum -> (n,n)
        diff = points[:, xp.newaxis, :] - points[xp.newaxis, :, :]
        return (diff**2).sum(axis=2)
    return fn


def heat_equation_step(xp, size):
    """One step of 1D heat equation: u_new = u + dt * d2u/dx2."""
    n = size
    u = xp.array(np.sin(np.linspace(0, 2*np.pi, n)))
    dt = 0.0001
    dx2 = (2*np.pi / n)**2

    def fn():
        # Laplacian via diff: d2u = u[:-2] - 2*u[1:-1] + u[2:]
        d2u = xp.diff(u, n=2)
        u_interior = u[1:-1] + dt * d2u / dx2
        # Construct result with boundaries - requires concatenate
        return xp.concatenate([[u[0]], u_interior, [u[-1]]])
    return fn


def random_walk(xp, size):
    """Random walk: cumsum of random steps."""
    rng = xp.random.default_rng(42)

    def fn():
        steps = rng.choice([-1, 1], size=size)
        return xp.cumsum(steps)
    return fn


def black_scholes_paths(xp, size):
    """Black-Scholes option pricing paths."""
    S0 = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    dt = T / 252
    rng = xp.random.default_rng(42)
    sqrt_dt = np.sqrt(dt)
    drift = (r - 0.5 * sigma**2) * dt

    def fn():
        Z = rng.standard_normal(size)
        returns = xp.exp(drift + sigma * sqrt_dt * Z)
        return S0 * xp.cumprod(returns)
    return fn


def vector_field_magnitude(xp, size):
    """Compute magnitude of a 2D vector field."""
    n = int(np.sqrt(size))
    vx = xp.array(np.random.randn(n, n))
    vy = xp.array(np.random.randn(n, n))

    def fn():
        return xp.sqrt(vx**2 + vy**2)
    return fn


def nbody_acceleration(xp, size):
    """N-body gravitational acceleration calculation."""
    n = int(np.sqrt(size))
    # Positions and masses
    pos = xp.array(np.random.randn(n, 3))
    mass = xp.array(np.random.rand(n) + 0.1)

    def fn():
        # Pairwise displacement: r_ij = pos_j - pos_i
        # Shape: (n, 1, 3) - (1, n, 3) -> (n, n, 3)
        r = pos[xp.newaxis, :, :] - pos[:, xp.newaxis, :]
        # Distance: |r_ij|
        dist_sq = (r**2).sum(axis=2)
        dist_sq = xp.maximum(dist_sq, 1e-10)  # avoid division by zero
        dist = xp.sqrt(dist_sq)
        # Acceleration: a_i = sum_j m_j * r_ij / |r_ij|^3
        inv_dist3 = 1.0 / (dist_sq * dist)
        # mass broadcast: (1, n) * (n, n, 3) -> need einsum or manual
        acc = (mass[xp.newaxis, :, xp.newaxis] * r * inv_dist3[:, :, xp.newaxis]).sum(axis=1)
        return acc
    return fn


def finite_difference_laplacian(xp, size):
    """2D Laplacian via finite differences."""
    n = int(np.sqrt(size))
    u = xp.array(np.random.randn(n, n))

    def fn():
        # 5-point stencil: laplacian = u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]
        lap = (u[2:, 1:-1] + u[:-2, 1:-1] +
               u[1:-1, 2:] + u[1:-1, :-2] -
               4 * u[1:-1, 1:-1])
        return lap
    return fn


BENCHMARKS = [
    ("Monte Carlo pi", monte_carlo_pi),  # requires xp.random.default_rng
    ("pairwise distances", pairwise_distances),  # requires xp.newaxis
    ("heat equation step", heat_equation_step),  # requires xp.concatenate with scalars
    ("random walk", random_walk),  # requires rng.choice, xp.cumsum
    ("Black-Scholes paths", black_scholes_paths),  # requires xp.cumprod
    ("vector field magnitude", vector_field_magnitude),
    ("N-body acceleration", nbody_acceleration),  # requires xp.newaxis, advanced indexing
    ("finite difference laplacian", finite_difference_laplacian),
]
