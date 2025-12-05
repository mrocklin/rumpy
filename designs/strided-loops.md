# Strided Inner Loops

NumPy-style inner loops that handle both contiguous and strided arrays.

## Core Insight

NumPy's inner loop signature passes stride information to the loop itself:
```c
void (*loop)(char **args, npy_intp *dimensions, npy_intp *steps, void *data)
```

The loop iterates N elements, advancing pointers by their strides. This handles
contiguous (stride=itemsize) and strided (stride!=itemsize) uniformly.

## Loop Signature

```rust
type BinaryLoopFn = unsafe fn(
    a: *const u8, b: *const u8, out: *mut u8,
    n: usize,
    strides: (isize, isize, isize),  // (a_stride, b_stride, out_stride)
);

type UnaryLoopFn = unsafe fn(
    src: *const u8, out: *mut u8,
    n: usize,
    strides: (isize, isize),  // (src_stride, out_stride)
);
```

## Contiguous Fast Path

Loops can check for contiguous case internally:
```rust
unsafe fn add_f64(a: *const u8, b: *const u8, out: *mut u8, n: usize, strides: (isize, isize, isize)) {
    let itemsize = 8isize;
    if strides == (itemsize, itemsize, itemsize) {
        // Contiguous: tight loop LLVM can vectorize
        let a = std::slice::from_raw_parts(a as *const f64, n);
        let b = std::slice::from_raw_parts(b as *const f64, n);
        let out = std::slice::from_raw_parts_mut(out as *mut f64, n);
        for i in 0..n {
            out[i] = a[i] + b[i];
        }
    } else {
        // Strided: pointer arithmetic
        let mut ap = a as *const f64;
        let mut bp = b as *const f64;
        let mut op = out as *mut f64;
        for _ in 0..n {
            *op = *ap + *bp;
            ap = (ap as *const u8).offset(strides.0) as *const f64;
            bp = (bp as *const u8).offset(strides.1) as *const f64;
            op = (op as *mut u8).offset(strides.2) as *mut f64;
        }
    }
}
```

## Benefits

1. **Single dispatch path** - no contiguous/strided branching in caller
2. **Loop controls optimization** - can have internal fast paths
3. **Matches NumPy** - familiar pattern, proven design
4. **Future SIMD** - easy to add explicit SIMD for contiguous case

## Registration

Same registry pattern, just different function signature:
```rust
registry.register_binary(BinaryOp::Add, DTypeKind::Float64, add_f64);
```

Macros generate loops to reduce boilerplate.
