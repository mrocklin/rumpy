---
name: numpy-api-compatibility
description: use when adding API to this project that exists in numpy, especially if the user mentions a stream in the numpy-api-parity plan.
---

When implementing functionality in this project that already exists in numpy  we can go through a few phases:

## Phase 1: Research and Design

We can look at how numpy implements a feature.  For simple functions we can look at the source code in ../numpy-src or by running simple scripts.

For more complex systems we can search the internet for NEPs.

We can also look through our own design files for internal systems in our codebase that we might use.  Often we find that `designs/adding-operations.md`, `designs/iteration-performance.md`, `designs/kernel-dispatch.md` are quite useful.

Check dtype behavior early. NumPy often preserves input dtypes - for example a float32 input often produces float32 output.

## Phase 2: Testing

We want simple and fast tests comparing execution against numpy.  Read `designs/testing.md`

## Phase 3: Implementation

Here we implement our new functionality, testing against our simple tests from earlier to gauge correctness.

**Avoid `to_vec()` in hot paths.** It converts all data to `Vec<f64>`, causing:
- Unnecessary memory allocation and copying
- dtype conversion overhead (especially f32→f64→f32)
- Wrong results for dtype-dependent operations (like `spacing`/ULP)

**Use typed dispatch instead.** Match on dtype and consider using typed pointer access:
```rust
match arr.dtype().kind() {
    DTypeKind::Float32 => process_typed::<f32>(arr, DType::float32()),
    DTypeKind::Float64 => process_typed::<f64>(arr, DType::float64()),
    _ => { /* convert or error */ }
}

fn process_typed<T: Float>(arr: &RumpyArray, dtype: DType) -> RumpyArray {
    let src_ptr = arr.data_ptr() as *const T;
    let result_ptr = result_buffer.as_mut_ptr() as *mut T;
    for i in 0..size {
        let x = unsafe { *src_ptr.add(i) };
        unsafe { *result_ptr.add(i) = /* compute */ };
    }
}
```

For generic math operations, define a trait (like `Float` with `sin()`, `exp()`, etc.) that f32 and f64 both implement. This lets generic code monomorphize to efficient type-specific code

## Phase 4: Simplification and Cleanup

Run the `/cleanup` command to review our work and see if there is anything we can simplify or clean up.

## Phase 5: Performance testing

Compare our performance against numpy.  If we're slower, we don't want to build fast-paths for specific dtypes (like float64) but instead want to build good dtype-generic solutions that compile down to something fast, often using SIMD operations.  We've managed this for other operations (even complex operations like reductions and norms), and numpy manages to do so, so we should have confidence that we can too.

If our operations don't require speed (if they just manage metadata) then this isn't a big deal.

**Always benchmark in release mode:**
```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv tool run maturin develop --release
```

**Benchmark across dtypes.** A common anti-pattern is code that's fast for float64 but slow for float32 or int due to hidden conversions:

If float32 is significantly slower than float64 (relative to numpy), suspect dtype conversion in the implementation.

## Phase 6: Cleanup again

Run the `/cleanup` operation again.

## Phase 7: Final review and commit

Present results about both what we've accomplished and performance if necessary.
