"""Micro-benchmarks to test performance hypotheses."""

# Hypothesis 1: Simple element-wise ops lack SIMD
def bench_simple_add(lib, size):
    """Pure element-wise add - should be SIMD-friendly."""
    a = lib.arange(size, dtype='float64')
    b = lib.arange(size, dtype='float64')
    return lambda: a + b

def bench_simple_mul(lib, size):
    """Pure element-wise multiply."""
    a = lib.arange(size, dtype='float64')
    b = lib.arange(size, dtype='float64')
    return lambda: a * b

def bench_simple_compare(lib, size):
    """Element-wise comparison (used in threshold)."""
    a = lib.arange(size, dtype='float64')
    return lambda: a > (size / 2)

def bench_scalar_mul(lib, size):
    """Scalar multiplication."""
    a = lib.arange(size, dtype='float64')
    return lambda: a * 2.5

def bench_scalar_add(lib, size):
    """Scalar addition."""
    a = lib.arange(size, dtype='float64')
    return lambda: a + 2.5


# Hypothesis 2: Temporary array allocation overhead
def bench_one_op(lib, size):
    """Single operation - baseline."""
    a = lib.arange(size, dtype='float64')
    return lambda: a + 1.0

def bench_two_ops(lib, size):
    """Two chained ops - 1 temporary."""
    a = lib.arange(size, dtype='float64')
    return lambda: (a + 1.0) * 2.0

def bench_three_ops(lib, size):
    """Three chained ops - 2 temporaries."""
    a = lib.arange(size, dtype='float64')
    return lambda: ((a + 1.0) * 2.0) - 0.5

def bench_four_ops(lib, size):
    """Four chained ops - 3 temporaries."""
    a = lib.arange(size, dtype='float64')
    return lambda: (((a + 1.0) * 2.0) - 0.5) / 1.5

def bench_six_ops(lib, size):
    """Six chained ops - 5 temporaries."""
    a = lib.arange(size, dtype='float64')
    return lambda: ((((((a + 1.0) * 2.0) - 0.5) / 1.5) + 3.0) * 0.5)


# Hypothesis 3: Reduction vs element-wise
def bench_sum_reduction(lib, size):
    """Full reduction - should be fast."""
    a = lib.arange(size, dtype='float64')
    return lambda: a.sum()

def bench_mean_reduction(lib, size):
    """Mean reduction."""
    a = lib.arange(size, dtype='float64')
    return lambda: a.mean()

def bench_max_reduction(lib, size):
    """Max reduction."""
    a = lib.arange(size, dtype='float64')
    return lambda: a.max()


# Hypothesis 4: Unary math functions
def bench_sqrt(lib, size):
    """Square root - transcendental function."""
    a = lib.arange(1, size + 1, dtype='float64')
    return lambda: lib.sqrt(a)

def bench_exp(lib, size):
    """Exponential."""
    a = lib.arange(size, dtype='float64') / size
    return lambda: lib.exp(a)

def bench_log(lib, size):
    """Natural log."""
    a = lib.arange(1, size + 1, dtype='float64')
    return lambda: lib.log(a)

def bench_sin(lib, size):
    """Sine function."""
    a = lib.arange(size, dtype='float64') / size
    return lambda: lib.sin(a)


# Hypothesis 5: Memory access patterns
def bench_contiguous_read(lib, size):
    """Read contiguous array."""
    a = lib.arange(size, dtype='float64')
    return lambda: a + 1.0

def bench_strided_read(lib, size):
    """Read every other element (stride=2)."""
    a = lib.arange(size * 2, dtype='float64')[::2]
    return lambda: a + 1.0

def bench_reversed_read(lib, size):
    """Read in reverse order."""
    a = lib.arange(size, dtype='float64')[::-1]
    return lambda: a + 1.0


# Hypothesis 6: Small vs large arrays (allocation dominance)
# Use different size ranges for this


# Hypothesis 7: dtype impact
def bench_f64_add(lib, size):
    """f64 addition."""
    a = lib.arange(size, dtype='float64')
    b = lib.arange(size, dtype='float64')
    return lambda: a + b

def bench_f32_add(lib, size):
    """f32 addition - 2x data density for SIMD."""
    a = lib.arange(size, dtype='float32')
    b = lib.arange(size, dtype='float32')
    return lambda: a + b

def bench_i64_add(lib, size):
    """i64 addition."""
    a = lib.arange(size, dtype='int64')
    b = lib.arange(size, dtype='int64')
    return lambda: a + b

def bench_i32_add(lib, size):
    """i32 addition."""
    a = lib.arange(size, dtype='int32')
    b = lib.arange(size, dtype='int32')
    return lambda: a + b


# Hypothesis 8: Broadcasting overhead
def bench_no_broadcast(lib, size):
    """Same shape, no broadcast needed."""
    a = lib.arange(size, dtype='float64').reshape((100, size // 100))
    b = lib.arange(size, dtype='float64').reshape((100, size // 100))
    return lambda: a + b

def bench_broadcast_scalar(lib, size):
    """Broadcast scalar to array."""
    a = lib.arange(size, dtype='float64')
    return lambda: a + 1.0

def bench_broadcast_row(lib, size):
    """Broadcast row across 2D array."""
    n = int(size ** 0.5)
    a = lib.arange(n * n, dtype='float64').reshape((n, n))
    b = lib.arange(n, dtype='float64').reshape((1, n))
    return lambda: a + b

def bench_broadcast_col(lib, size):
    """Broadcast column across 2D array."""
    n = int(size ** 0.5)
    a = lib.arange(n * n, dtype='float64').reshape((n, n))
    b = lib.arange(n, dtype='float64').reshape((n, 1))
    return lambda: a + b


BENCHMARKS = [
    # Hypothesis 1: SIMD
    ("H1: simple add", bench_simple_add),
    ("H1: simple mul", bench_simple_mul),
    ("H1: simple compare", bench_simple_compare),
    ("H1: scalar mul", bench_scalar_mul),
    ("H1: scalar add", bench_scalar_add),

    # Hypothesis 2: Temporaries
    ("H2: 1 op (baseline)", bench_one_op),
    ("H2: 2 ops (1 temp)", bench_two_ops),
    ("H2: 3 ops (2 temps)", bench_three_ops),
    ("H2: 4 ops (3 temps)", bench_four_ops),
    ("H2: 6 ops (5 temps)", bench_six_ops),

    # Hypothesis 3: Reductions
    ("H3: sum reduction", bench_sum_reduction),
    ("H3: mean reduction", bench_mean_reduction),
    ("H3: max reduction", bench_max_reduction),

    # Hypothesis 4: Unary math
    ("H4: sqrt", bench_sqrt),
    ("H4: exp", bench_exp),
    ("H4: log", bench_log),
    ("H4: sin", bench_sin),

    # Hypothesis 5: Memory patterns
    ("H5: contiguous", bench_contiguous_read),
    ("H5: strided (2)", bench_strided_read),
    ("H5: reversed", bench_reversed_read),

    # Hypothesis 7: dtype
    ("H7: f64 add", bench_f64_add),
    ("H7: f32 add", bench_f32_add),
    ("H7: i64 add", bench_i64_add),
    ("H7: i32 add", bench_i32_add),

    # Hypothesis 8: Broadcasting
    ("H8: no broadcast", bench_no_broadcast),
    ("H8: broadcast scalar", bench_broadcast_scalar),
    ("H8: broadcast row", bench_broadcast_row),
    ("H8: broadcast col", bench_broadcast_col),
]
