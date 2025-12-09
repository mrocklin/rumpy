"""Quick I/O benchmarks - smaller sizes for fast feedback."""
import tempfile
import time
import os
import numpy as np
import rumpy as rp


def bench(name, fn, runs=3):
    """Simple benchmark."""
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def fmt(sec):
    if sec < 1e-3:
        return f"{sec*1e6:.0f}µs"
    elif sec < 1:
        return f"{sec*1e3:.1f}ms"
    return f"{sec:.2f}s"


def ratio(np_t, rp_t):
    r = np_t / rp_t
    if r > 1.2:
        return f"{r:.1f}x faster"
    elif r < 0.8:
        return f"{1/r:.1f}x slower"
    return f"~same"


print("=== Quick I/O Benchmarks ===\n")

# Binary save/load (most common, fast)
print("Binary .npy (10K float64):")
n = np.random.rand(10000).astype(np.float64)
r = rp.asarray(n)
with tempfile.TemporaryDirectory() as tmp:
    np_f, rp_f = f"{tmp}/n.npy", f"{tmp}/r.npy"
    np_t = bench("np save", lambda: np.save(np_f, n))
    rp_t = bench("rp save", lambda: rp.save(rp_f, r))
    print(f"  save: numpy={fmt(np_t)}, rumpy={fmt(rp_t)}, {ratio(np_t, rp_t)}")
    np.save(np_f, n); rp.save(rp_f, r)
    np_t = bench("np load", lambda: np.load(np_f))
    rp_t = bench("rp load", lambda: rp.load(rp_f))
    print(f"  load: numpy={fmt(np_t)}, rumpy={fmt(rp_t)}, {ratio(np_t, rp_t)}")

# frombuffer (should be fast - just pointer wrapping)
print("\nfrombuffer (10K float64):")
buf = n.tobytes()
np_t = bench("np", lambda: np.frombuffer(buf, dtype=np.float64))
rp_t = bench("rp", lambda: rp.frombuffer(buf, dtype="float64"))
print(f"  numpy={fmt(np_t)}, rumpy={fmt(rp_t)}, {ratio(np_t, rp_t)}")

# fromfile binary
print("\nfromfile binary (10K float64):")
with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
    n.tofile(f.name)
    fname = f.name
np_t = bench("np", lambda: np.fromfile(fname, dtype=np.float64))
rp_t = bench("rp", lambda: rp.fromfile(fname, dtype="float64"))
print(f"  numpy={fmt(np_t)}, rumpy={fmt(rp_t)}, {ratio(np_t, rp_t)}")
os.unlink(fname)

# tofile binary
print("\ntofile binary (10K float64):")
with tempfile.TemporaryDirectory() as tmp:
    np_f, rp_f = f"{tmp}/n.bin", f"{tmp}/r.bin"
    np_t = bench("np", lambda: n.tofile(np_f))
    rp_t = bench("rp", lambda: r.tofile(rp_f))
    print(f"  numpy={fmt(np_t)}, rumpy={fmt(rp_t)}, {ratio(np_t, rp_t)}")

# Text I/O (slower, use smaller size)
print("\nloadtxt (1K rows x 3 cols):")
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    for i in range(1000):
        f.write(f"{i*1.1} {i*2.2} {i*3.3}\n")
    fname = f.name
np_t = bench("np", lambda: np.loadtxt(fname))
rp_t = bench("rp", lambda: rp.loadtxt(fname))
print(f"  numpy={fmt(np_t)}, rumpy={fmt(rp_t)}, {ratio(np_t, rp_t)}")
os.unlink(fname)

print("\nsavetxt (1K rows x 3 cols):")
n2 = np.random.rand(1000, 3)
r2 = rp.asarray(n2)
with tempfile.TemporaryDirectory() as tmp:
    np_f, rp_f = f"{tmp}/n.txt", f"{tmp}/r.txt"
    np_t = bench("np", lambda: np.savetxt(np_f, n2))
    rp_t = bench("rp", lambda: rp.savetxt(rp_f, r2))
    print(f"  numpy={fmt(np_t)}, rumpy={fmt(rp_t)}, {ratio(np_t, rp_t)}")

# NPZ
print("\nsavez (two 1K arrays):")
n1 = np.random.rand(1000).astype(np.float64)
n2 = np.random.rand(1000, 10).astype(np.float64)
r1, r2 = rp.asarray(n1), rp.asarray(n2)
with tempfile.TemporaryDirectory() as tmp:
    np_f, rp_f = f"{tmp}/n.npz", f"{tmp}/r.npz"
    np_t = bench("np", lambda: np.savez(np_f, a=n1, b=n2))
    rp_t = bench("rp", lambda: rp.savez(rp_f, a=r1, b=r2))
    print(f"  numpy={fmt(np_t)}, rumpy={fmt(rp_t)}, {ratio(np_t, rp_t)}")

print("\n=== Done ===")
