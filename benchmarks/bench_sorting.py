"""Benchmarks for sorting operations."""


def bench_sort_1d(lib, size):
    """Full sort of 1D array."""
    # Random-ish data to avoid best/worst case
    a = lib.arange(size, dtype='float64')
    a = a * 7 % size  # Pseudo-shuffle
    return lambda: lib.sort(a)


def bench_argsort_1d(lib, size):
    """Argsort of 1D array."""
    a = lib.arange(size, dtype='float64')
    a = a * 7 % size
    return lambda: lib.argsort(a)


def bench_partition_1d(lib, size):
    """Partition (partial sort) of 1D array - median position."""
    a = lib.arange(size, dtype='float64')
    a = a * 7 % size
    kth = size // 2
    return lambda: lib.partition(a, kth)


def bench_argpartition_1d(lib, size):
    """Argpartition of 1D array - median position."""
    a = lib.arange(size, dtype='float64')
    a = a * 7 % size
    kth = size // 2
    return lambda: lib.argpartition(a, kth)


def bench_partition_small_k(lib, size):
    """Partition with small k (finding min elements)."""
    a = lib.arange(size, dtype='float64')
    a = a * 7 % size
    kth = 10  # Find 10 smallest
    return lambda: lib.partition(a, kth)


def bench_lexsort_2keys(lib, size):
    """Lexsort with 2 keys."""
    k1 = lib.arange(size, dtype='float64') % 100
    k2 = lib.arange(size, dtype='float64') % 10
    return lambda: lib.lexsort((k1, k2))


def bench_lexsort_3keys(lib, size):
    """Lexsort with 3 keys."""
    k1 = lib.arange(size, dtype='float64') % 100
    k2 = lib.arange(size, dtype='float64') % 10
    k3 = lib.arange(size, dtype='float64') % 5
    return lambda: lib.lexsort((k1, k2, k3))


def bench_sort_2d_axis1(lib, size):
    """Sort 2D array along axis 1."""
    n = int(size ** 0.5)
    a = lib.arange(n * n, dtype='float64').reshape((n, n))
    a = a * 7 % (n * n)
    return lambda: lib.sort(a, axis=1)


def bench_sort_2d_axis0(lib, size):
    """Sort 2D array along axis 0."""
    n = int(size ** 0.5)
    a = lib.arange(n * n, dtype='float64').reshape((n, n))
    a = a * 7 % (n * n)
    return lambda: lib.sort(a, axis=0)


BENCHMARKS = [
    ("sort 1D", bench_sort_1d),
    ("argsort 1D", bench_argsort_1d),
    ("partition 1D (k=n/2)", bench_partition_1d),
    ("argpartition 1D (k=n/2)", bench_argpartition_1d),
    ("partition 1D (k=10)", bench_partition_small_k),
    ("lexsort 2 keys", bench_lexsort_2keys),
    ("lexsort 3 keys", bench_lexsort_3keys),
    ("sort 2D axis=1", bench_sort_2d_axis1),
    ("sort 2D axis=0", bench_sort_2d_axis0),
]
