"""Benchmarks for Stream 14 linear algebra functions."""
import time
import numpy as np
import rumpy as rp

def bench(name, rp_fn, np_fn, warmup=3, runs=10):
    """Benchmark rumpy vs numpy."""
    # Warmup
    for _ in range(warmup):
        rp_fn()
        np_fn()

    # Time rumpy
    start = time.perf_counter()
    for _ in range(runs):
        rp_fn()
    rp_time = (time.perf_counter() - start) / runs * 1000

    # Time numpy
    start = time.perf_counter()
    for _ in range(runs):
        np_fn()
    np_time = (time.perf_counter() - start) / runs * 1000

    ratio = rp_time / np_time
    status = "✓" if ratio < 2.0 else "⚠" if ratio < 5.0 else "✗"
    print(f"{status} {name:30} rp={rp_time:8.3f}ms  np={np_time:8.3f}ms  ratio={ratio:.2f}x")
    return rp_time, np_time

print("=" * 80)
print("Stream 14 Linear Algebra Benchmarks (release build)")
print("=" * 80)

# Small matrices for decomposition tests
sizes = [(50, 50), (100, 100), (200, 200)]

for m, n in sizes:
    print(f"\n--- Matrix size: {m}x{n} ---")

    # Create test data
    A_np = np.random.randn(m, n)
    A_rp = rp.asarray(A_np)

    # Square matrix for eigenvalue tests
    if m == n:
        # Use default args to capture by value, not by reference
        bench(f"eig ({m}x{n})",
              lambda a=A_rp: rp.linalg.eig(a),
              lambda a=A_np: np.linalg.eig(a))

        bench(f"eigvals ({m}x{n})",
              lambda a=A_rp: rp.linalg.eigvals(a),
              lambda a=A_np: np.linalg.eigvals(a))

        bench(f"slogdet ({m}x{n})",
              lambda a=A_rp: rp.linalg.slogdet(a),
              lambda a=A_np: np.linalg.slogdet(a))

        bench(f"cond ({m}x{n})",
              lambda a=A_rp: rp.linalg.cond(a),
              lambda a=A_np: np.linalg.cond(a))

        bench(f"pinv ({m}x{n})",
              lambda a=A_rp: rp.linalg.pinv(a),
              lambda a=A_np: np.linalg.pinv(a))

        bench(f"matrix_rank ({m}x{n})",
              lambda a=A_rp: rp.linalg.matrix_rank(a),
              lambda a=A_np: np.linalg.matrix_rank(a))

# lstsq - overdetermined system
print(f"\n--- Least squares ---")
for m, n in [(100, 50), (500, 100), (1000, 100)]:
    A_np = np.random.randn(m, n)
    b_np = np.random.randn(m)
    A_rp = rp.asarray(A_np)
    b_rp = rp.asarray(b_np)

    bench(f"lstsq ({m}x{n})",
          lambda a=A_rp, b=b_rp: rp.linalg.lstsq(a, b),
          lambda a=A_np, b=b_np: np.linalg.lstsq(a, b, rcond=None))

# Module-level functions
print(f"\n--- Module-level functions ---")

# vdot - various sizes
for size in [100, 1000, 10000]:
    a_np = np.random.randn(size)
    b_np = np.random.randn(size)
    a_rp = rp.asarray(a_np)
    b_rp = rp.asarray(b_np)

    bench(f"vdot (n={size})",
          lambda a=a_rp, b=b_rp: rp.vdot(a, b),
          lambda a=a_np, b=b_np: np.vdot(a, b))

# kron
for size in [10, 20, 50]:
    a_np = np.random.randn(size, size)
    b_np = np.random.randn(size, size)
    a_rp = rp.asarray(a_np)
    b_rp = rp.asarray(b_np)

    bench(f"kron ({size}x{size} ⊗ {size}x{size})",
          lambda a=a_rp, b=b_rp: rp.kron(a, b),
          lambda a=a_np, b=b_np: np.kron(a, b))

# cross
a_np = np.random.randn(3)
b_np = np.random.randn(3)
a_rp = rp.asarray(a_np)
b_rp = rp.asarray(b_np)

bench("cross (3-vectors)",
      lambda a=a_rp, b=b_rp: rp.cross(a, b),
      lambda a=a_np, b=b_np: np.cross(a, b),
      runs=1000)

# tensordot
for size in [50, 100, 200]:
    a_np = np.random.randn(size, size)
    b_np = np.random.randn(size, size)
    a_rp = rp.asarray(a_np)
    b_rp = rp.asarray(b_np)

    bench(f"tensordot ({size}x{size}, axes=1)",
          lambda a=a_rp, b=b_rp: rp.tensordot(a, b, 1),
          lambda a=a_np, b=b_np: np.tensordot(a, b, 1))

print("\n" + "=" * 80)
print("Legend: ✓ < 2x slower, ⚠ 2-5x slower, ✗ > 5x slower")
print("=" * 80)
