"""Linear algebra benchmarks - realistic workflows.

Each benchmark is a function that takes (xp, size) where xp is numpy or rumpy,
and returns a callable that performs the benchmark.
"""

import numpy as np


def kalman_predict(xp, size):
    """Kalman filter predict step: x = F @ x, P = F @ P @ F.T + Q"""
    n = max(int(np.sqrt(size)), 4)
    F = xp.array(np.eye(n) + 0.1 * np.random.randn(n, n))
    Q = xp.array(np.eye(n) * 0.01)
    x = xp.array(np.random.randn(n, 1))
    P = xp.array(np.eye(n))

    def fn():
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        return x_pred, P_pred
    return fn


def kalman_update(xp, size):
    """Kalman filter update step with measurement."""
    n = max(int(np.sqrt(size)), 4)
    m = n // 2
    H = xp.array(np.random.randn(m, n))
    R = xp.array(np.eye(m) * 0.1)
    x = xp.array(np.random.randn(n, 1))
    P = xp.array(np.eye(n))
    z = xp.array(np.random.randn(m, 1))
    I = xp.eye(n)

    def fn():
        y = z - H @ x
        S = H @ P @ H.T + R
        # K = P @ H.T @ inv(S), but we use solve for stability
        K = xp.linalg.solve(S.T, (P @ H.T).T).T
        x_new = x + K @ y
        P_new = (I - K @ H) @ P
        return x_new, P_new
    return fn


def least_squares_svd(xp, size):
    """Least squares via SVD: solve Ax = b."""
    m = int(np.sqrt(size))
    n = m // 2
    A = xp.array(np.random.randn(m, n))
    b = xp.array(np.random.randn(m))

    def fn():
        U, s, Vt = xp.linalg.svd(A, full_matrices=False)
        # x = V @ diag(1/s) @ U.T @ b
        return Vt.T @ (xp.diag(1.0 / s) @ (U.T @ b))
    return fn


def qr_solve(xp, size):
    """Solve Ax = b via QR decomposition."""
    n = int(np.sqrt(size))
    A = xp.array(np.random.randn(n, n))
    b = xp.array(np.random.randn(n))

    def fn():
        Q, R = xp.linalg.qr(A)
        return xp.linalg.solve(R, Q.T @ b)
    return fn


def matrix_power(xp, size):
    """Compute A^4 via repeated multiplication."""
    n = int(np.sqrt(size))
    A = xp.array(np.random.randn(n, n) / n)

    def fn():
        A2 = A @ A
        return A2 @ A2
    return fn


def gram_schmidt(xp, size):
    """Gram-Schmidt orthogonalization (QR)."""
    n = int(np.sqrt(size))
    A = xp.array(np.random.randn(n, n))

    def fn():
        Q, R = xp.linalg.qr(A)
        return Q
    return fn


def eigendecomposition(xp, size):
    """Eigendecomposition of symmetric matrix."""
    n = int(np.sqrt(size))
    A_np = np.random.randn(n, n)
    A_np = (A_np + A_np.T) / 2
    A = xp.array(A_np)

    def fn():
        return xp.linalg.eigh(A)
    return fn


def cholesky_solve(xp, size):
    """Solve via Cholesky decomposition for SPD matrix."""
    n = int(np.sqrt(size))
    A_np = np.random.randn(n, n)
    A_np = A_np @ A_np.T + np.eye(n)  # SPD
    A = xp.array(A_np)
    b = xp.array(np.random.randn(n))

    def fn():
        L = xp.linalg.cholesky(A)
        # Solve L @ L.T @ x = b
        y = xp.linalg.solve(L, b)
        return xp.linalg.solve(L.T, y)
    return fn


def matrix_norm(xp, size):
    """Compute Frobenius norm."""
    n = int(np.sqrt(size))
    A = xp.array(np.random.randn(n, n))

    def fn():
        return xp.linalg.norm(A, 'fro')
    return fn


BENCHMARKS = [
    ("Kalman predict", kalman_predict),
    ("Kalman update", kalman_update),  # requires xp.linalg.solve
    ("least squares (SVD)", least_squares_svd),  # requires xp.linalg.svd, xp.diag
    ("QR solve", qr_solve),  # requires xp.linalg.qr, xp.linalg.solve
    ("matrix power (A^4)", matrix_power),
    ("Gram-Schmidt (QR)", gram_schmidt),
    ("eigendecomposition", eigendecomposition),  # requires xp.linalg.eigh
    ("Cholesky solve", cholesky_solve),  # requires xp.linalg.cholesky
    ("matrix norm", matrix_norm),  # requires xp.linalg.norm
]
