//! UFunc registry for type-specific inner loops.
//!
//! Most binary/unary ops are now handled by the kernel/dispatch system.
//! This registry is kept for:
//! - Reduce loops (full-array and axis reductions)
//! - Bitwise operations

use crate::array::dtype::{BitwiseOp, DTypeKind, ReduceOp};
use half::f16;
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

/// Type signature for a ufunc loop.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TypeSignature {
    pub inputs: Vec<DTypeKind>,
    pub output: DTypeKind,
}

impl TypeSignature {
    pub fn reduce(input: DTypeKind, out: DTypeKind) -> Self {
        Self {
            inputs: vec![input],
            output: out,
        }
    }
}

/// Inner loop function for binary operations.
///
/// Processes `n` elements, advancing pointers by strides.
/// The loop handles both contiguous (stride=itemsize) and strided cases.
///
/// # Safety
/// Caller must ensure pointers are valid for `n` elements with given strides.
pub type BinaryLoopFn = unsafe fn(
    a_ptr: *const u8,
    b_ptr: *const u8,
    out_ptr: *mut u8,
    n: usize,
    strides: (isize, isize, isize),  // (a_stride, b_stride, out_stride)
);

/// Inner loop function for unary operations.
///
/// Processes `n` elements, advancing pointers by strides.
///
/// # Safety
/// Caller must ensure pointers are valid for `n` elements with given strides.
pub type UnaryLoopFn = unsafe fn(
    src_ptr: *const u8,
    out_ptr: *mut u8,
    n: usize,
    strides: (isize, isize),  // (src_stride, out_stride)
);

/// Initialize accumulator for reduction.
pub type ReduceInitFn = unsafe fn(out_ptr: *mut u8, idx: usize);

/// Accumulate one element into reduction (legacy, for full-array reductions).
pub type ReduceAccFn = unsafe fn(
    acc_ptr: *mut u8,
    idx: usize,
    val_ptr: *const u8,
    byte_offset: isize,
);

/// Strided reduction loop - processes N elements into an accumulator.
///
/// This is the fast path for axis reductions. Handles both contiguous
/// (stride == itemsize) and strided cases with SIMD-friendly inner loops.
///
/// # Safety
/// Caller must ensure:
/// - `acc_ptr` points to valid, initialized accumulator
/// - `src_ptr` is valid for `n` elements at `stride` intervals
pub type ReduceLoopFn = unsafe fn(
    acc_ptr: *mut u8,      // Single accumulator element
    src_ptr: *const u8,    // Source data start
    n: usize,              // Number of elements to reduce
    stride: isize,         // Byte stride between elements
);

/// Registry for ufunc inner loops.
///
/// Note: Binary and unary loops are now handled by the kernel/dispatch system
/// in `dispatch.rs`. This registry is kept for:
/// - Reduce loops (full-array reductions)
/// - Reduce strided loops (axis reductions)
/// - Bitwise operations
pub struct UFuncRegistry {
    reduce_loops: HashMap<(ReduceOp, TypeSignature), (ReduceInitFn, ReduceAccFn)>,
    /// Strided reduction loops for axis reductions (fast path)
    reduce_strided_loops: HashMap<(ReduceOp, TypeSignature), (ReduceInitFn, ReduceLoopFn)>,
    /// Bitwise binary loops (integer/bool types)
    bitwise_binary_loops: HashMap<(BitwiseOp, DTypeKind), BinaryLoopFn>,
    /// Bitwise NOT loops (integer/bool types)
    bitwise_not_loops: HashMap<DTypeKind, UnaryLoopFn>,
}

impl UFuncRegistry {
    pub fn new() -> Self {
        Self {
            reduce_loops: HashMap::new(),
            reduce_strided_loops: HashMap::new(),
            bitwise_binary_loops: HashMap::new(),
            bitwise_not_loops: HashMap::new(),
        }
    }

    /// Register a reduce loop for a specific operation and type signature.
    pub fn register_reduce(
        &mut self,
        op: ReduceOp,
        sig: TypeSignature,
        init: ReduceInitFn,
        acc: ReduceAccFn,
    ) {
        self.reduce_loops.insert((op, sig), (init, acc));
    }

    /// Look up a reduce loop. Returns None if no registered loop exists.
    pub fn lookup_reduce(
        &self,
        op: ReduceOp,
        input: DTypeKind,
    ) -> Option<(ReduceInitFn, ReduceAccFn, DTypeKind)> {
        let sig = TypeSignature::reduce(input.clone(), input.clone());
        self.reduce_loops.get(&(op, sig)).map(|&(init, acc)| (init, acc, input))
    }

    /// Register a strided reduce loop for axis reductions.
    pub fn register_reduce_strided(
        &mut self,
        op: ReduceOp,
        sig: TypeSignature,
        init: ReduceInitFn,
        loop_fn: ReduceLoopFn,
    ) {
        self.reduce_strided_loops.insert((op, sig), (init, loop_fn));
    }

    /// Look up a strided reduce loop. Returns None if no registered loop exists.
    pub fn lookup_reduce_strided(
        &self,
        op: ReduceOp,
        input: DTypeKind,
    ) -> Option<(ReduceInitFn, ReduceLoopFn, DTypeKind)> {
        let sig = TypeSignature::reduce(input.clone(), input.clone());
        self.reduce_strided_loops.get(&(op, sig)).map(|&(init, loop_fn)| (init, loop_fn, input))
    }

    /// Register a bitwise binary loop for a specific operation and dtype.
    pub fn register_bitwise_binary(&mut self, op: BitwiseOp, dtype: DTypeKind, f: BinaryLoopFn) {
        self.bitwise_binary_loops.insert((op, dtype), f);
    }

    /// Look up a bitwise binary loop.
    pub fn lookup_bitwise_binary(&self, op: BitwiseOp, dtype: DTypeKind) -> Option<BinaryLoopFn> {
        self.bitwise_binary_loops.get(&(op, dtype)).copied()
    }

    /// Register a bitwise NOT loop for a specific dtype.
    pub fn register_bitwise_not(&mut self, dtype: DTypeKind, f: UnaryLoopFn) {
        self.bitwise_not_loops.insert(dtype, f);
    }

    /// Look up a bitwise NOT loop.
    pub fn lookup_bitwise_not(&self, dtype: DTypeKind) -> Option<UnaryLoopFn> {
        self.bitwise_not_loops.get(&dtype).copied()
    }
}

impl Default for UFuncRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// Global registry
static REGISTRY: OnceLock<RwLock<UFuncRegistry>> = OnceLock::new();

/// Get reference to the global registry.
pub fn registry() -> &'static RwLock<UFuncRegistry> {
    REGISTRY.get_or_init(|| RwLock::new(init_default_loops()))
}

/// Initialize default loops for common types.
///
/// Note: Binary and unary loops have been moved to the kernel/dispatch system.
/// This registry now only contains:
/// - Reduce loops (full-array and axis reductions)
/// - Bitwise operations
fn init_default_loops() -> UFuncRegistry {
    let mut reg = UFuncRegistry::new();

    // ========================================================================
    // Reduce loops (for types not handled by kernel/dispatch system)
    // Dispatch handles: f64, f32, i64, i32, i16, u64, u32, u16, u8, complex128, complex64
    // Registry handles: Float16, Bool (and DateTime64 via trait fallback)
    // ========================================================================

    // Float16 reduce loops (convert to f32 for ops)
    reg.register_reduce(
        ReduceOp::Sum, TypeSignature::reduce(DTypeKind::Float16, DTypeKind::Float16),
        |out_ptr, idx| unsafe { *(out_ptr as *mut f16).add(idx) = f16::ZERO; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut f16).add(idx);
            let a = (*acc).to_f32();
            let v = (*(val_ptr.offset(byte_offset) as *const f16)).to_f32();
            *acc = f16::from_f32(a + v);
        },
    );
    reg.register_reduce(
        ReduceOp::Prod, TypeSignature::reduce(DTypeKind::Float16, DTypeKind::Float16),
        |out_ptr, idx| unsafe { *(out_ptr as *mut f16).add(idx) = f16::ONE; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut f16).add(idx);
            let a = (*acc).to_f32();
            let v = (*(val_ptr.offset(byte_offset) as *const f16)).to_f32();
            *acc = f16::from_f32(a * v);
        },
    );
    reg.register_reduce(
        ReduceOp::Max, TypeSignature::reduce(DTypeKind::Float16, DTypeKind::Float16),
        |out_ptr, idx| unsafe { *(out_ptr as *mut f16).add(idx) = f16::NEG_INFINITY; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut f16).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const f16);
            if v > *acc { *acc = v; }
        },
    );
    reg.register_reduce(
        ReduceOp::Min, TypeSignature::reduce(DTypeKind::Float16, DTypeKind::Float16),
        |out_ptr, idx| unsafe { *(out_ptr as *mut f16).add(idx) = f16::INFINITY; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut f16).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const f16);
            if v < *acc { *acc = v; }
        },
    );

    // Bool reduce loops (Sum=any, Prod=all)
    reg.register_reduce(
        ReduceOp::Sum,
        TypeSignature::reduce(DTypeKind::Bool, DTypeKind::Bool),
        |out_ptr, idx| unsafe { *out_ptr.add(idx) = 0; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = acc_ptr.add(idx);
            let v = *val_ptr.offset(byte_offset);
            if v != 0 { *acc = 1; }
        },
    );
    reg.register_reduce(
        ReduceOp::Prod,
        TypeSignature::reduce(DTypeKind::Bool, DTypeKind::Bool),
        |out_ptr, idx| unsafe { *out_ptr.add(idx) = 1; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = acc_ptr.add(idx);
            let v = *val_ptr.offset(byte_offset);
            if v == 0 { *acc = 0; }
        },
    );

    // ========================================================================
    // Bitwise operations (integer/bool types)
    // ========================================================================

    // Macro for bitwise binary loops with contiguous fast path
    macro_rules! register_bitwise_binary {
        ($reg:expr, $op:expr, $kind:expr, $T:ty, $f:expr) => {
            $reg.register_bitwise_binary(
                $op,
                $kind.clone(),
                |a_ptr, b_ptr, out_ptr, n, strides| unsafe {
                    let itemsize = std::mem::size_of::<$T>() as isize;
                    let (sa, sb, so) = strides;
                    if sa == itemsize && sb == itemsize && so == itemsize {
                        // Contiguous fast path - LLVM can auto-vectorize
                        let a = std::slice::from_raw_parts(a_ptr as *const $T, n);
                        let b = std::slice::from_raw_parts(b_ptr as *const $T, n);
                        let out = std::slice::from_raw_parts_mut(out_ptr as *mut $T, n);
                        for i in 0..n {
                            out[i] = $f(a[i], b[i]);
                        }
                    } else {
                        // Strided path
                        let mut ap = a_ptr;
                        let mut bp = b_ptr;
                        let mut op = out_ptr;
                        for _ in 0..n {
                            let a = *(ap as *const $T);
                            let b = *(bp as *const $T);
                            *(op as *mut $T) = $f(a, b);
                            ap = ap.offset(sa);
                            bp = bp.offset(sb);
                            op = op.offset(so);
                        }
                    }
                },
            );
        };
    }

    // Macro for bitwise NOT loops
    macro_rules! register_bitwise_not {
        ($reg:expr, $kind:expr, $T:ty) => {
            $reg.register_bitwise_not(
                $kind.clone(),
                |src_ptr, out_ptr, n, strides| unsafe {
                    let itemsize = std::mem::size_of::<$T>() as isize;
                    let (ss, so) = strides;
                    if ss == itemsize && so == itemsize {
                        // Contiguous fast path
                        let src = std::slice::from_raw_parts(src_ptr as *const $T, n);
                        let out = std::slice::from_raw_parts_mut(out_ptr as *mut $T, n);
                        for i in 0..n {
                            out[i] = !src[i];
                        }
                    } else {
                        // Strided path
                        let mut sp = src_ptr;
                        let mut op = out_ptr;
                        for _ in 0..n {
                            *(op as *mut $T) = !(*(sp as *const $T));
                            sp = sp.offset(ss);
                            op = op.offset(so);
                        }
                    }
                },
            );
        };
    }

    // Register all bitwise ops for an integer type
    macro_rules! register_bitwise_ops {
        ($reg:expr, $kind:expr, $T:ty) => {
            register_bitwise_binary!($reg, BitwiseOp::And, $kind, $T, |a: $T, b: $T| a & b);
            register_bitwise_binary!($reg, BitwiseOp::Or, $kind, $T, |a: $T, b: $T| a | b);
            register_bitwise_binary!($reg, BitwiseOp::Xor, $kind, $T, |a: $T, b: $T| a ^ b);
            register_bitwise_binary!($reg, BitwiseOp::LeftShift, $kind, $T, |a: $T, b: $T| a.wrapping_shl(b as u32));
            register_bitwise_binary!($reg, BitwiseOp::RightShift, $kind, $T, |a: $T, b: $T| a.wrapping_shr(b as u32));
            register_bitwise_not!($reg, $kind, $T);
        };
    }

    // Register for all integer types
    register_bitwise_ops!(reg, DTypeKind::Int64, i64);
    register_bitwise_ops!(reg, DTypeKind::Int32, i32);
    register_bitwise_ops!(reg, DTypeKind::Int16, i16);
    register_bitwise_ops!(reg, DTypeKind::Uint64, u64);
    register_bitwise_ops!(reg, DTypeKind::Uint32, u32);
    register_bitwise_ops!(reg, DTypeKind::Uint16, u16);
    register_bitwise_ops!(reg, DTypeKind::Uint8, u8);

    // Bool bitwise (use logical operators)
    reg.register_bitwise_binary(
        BitwiseOp::And,
        DTypeKind::Bool,
        |a_ptr, b_ptr, out_ptr, n, strides| unsafe {
            let (sa, sb, so) = strides;
            if sa == 1 && sb == 1 && so == 1 {
                let a = std::slice::from_raw_parts(a_ptr, n);
                let b = std::slice::from_raw_parts(b_ptr, n);
                let out = std::slice::from_raw_parts_mut(out_ptr, n);
                for i in 0..n {
                    out[i] = (a[i] != 0 && b[i] != 0) as u8;
                }
            } else {
                let (mut ap, mut bp, mut op) = (a_ptr, b_ptr, out_ptr);
                for _ in 0..n {
                    *op = (*ap != 0 && *bp != 0) as u8;
                    ap = ap.offset(sa);
                    bp = bp.offset(sb);
                    op = op.offset(so);
                }
            }
        },
    );
    reg.register_bitwise_binary(
        BitwiseOp::Or,
        DTypeKind::Bool,
        |a_ptr, b_ptr, out_ptr, n, strides| unsafe {
            let (sa, sb, so) = strides;
            if sa == 1 && sb == 1 && so == 1 {
                let a = std::slice::from_raw_parts(a_ptr, n);
                let b = std::slice::from_raw_parts(b_ptr, n);
                let out = std::slice::from_raw_parts_mut(out_ptr, n);
                for i in 0..n {
                    out[i] = (a[i] != 0 || b[i] != 0) as u8;
                }
            } else {
                let (mut ap, mut bp, mut op) = (a_ptr, b_ptr, out_ptr);
                for _ in 0..n {
                    *op = (*ap != 0 || *bp != 0) as u8;
                    ap = ap.offset(sa);
                    bp = bp.offset(sb);
                    op = op.offset(so);
                }
            }
        },
    );
    reg.register_bitwise_binary(
        BitwiseOp::Xor,
        DTypeKind::Bool,
        |a_ptr, b_ptr, out_ptr, n, strides| unsafe {
            let (sa, sb, so) = strides;
            if sa == 1 && sb == 1 && so == 1 {
                let a = std::slice::from_raw_parts(a_ptr, n);
                let b = std::slice::from_raw_parts(b_ptr, n);
                let out = std::slice::from_raw_parts_mut(out_ptr, n);
                for i in 0..n {
                    out[i] = ((a[i] != 0) != (b[i] != 0)) as u8;
                }
            } else {
                let (mut ap, mut bp, mut op) = (a_ptr, b_ptr, out_ptr);
                for _ in 0..n {
                    *op = ((*ap != 0) != (*bp != 0)) as u8;
                    ap = ap.offset(sa);
                    bp = bp.offset(sb);
                    op = op.offset(so);
                }
            }
        },
    );
    reg.register_bitwise_not(
        DTypeKind::Bool,
        |src_ptr, out_ptr, n, strides| unsafe {
            let (ss, so) = strides;
            if ss == 1 && so == 1 {
                let src = std::slice::from_raw_parts(src_ptr, n);
                let out = std::slice::from_raw_parts_mut(out_ptr, n);
                for i in 0..n {
                    out[i] = (src[i] == 0) as u8;
                }
            } else {
                let (mut sp, mut op) = (src_ptr, out_ptr);
                for _ in 0..n {
                    *op = (*sp == 0) as u8;
                    sp = sp.offset(ss);
                    op = op.offset(so);
                }
            }
        },
    );

    reg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_signature_equality() {
        let sig1 = TypeSignature::reduce(DTypeKind::Float64, DTypeKind::Float64);
        let sig2 = TypeSignature::reduce(DTypeKind::Float64, DTypeKind::Float64);
        let sig3 = TypeSignature::reduce(DTypeKind::Float32, DTypeKind::Float32);

        assert_eq!(sig1, sig2);
        assert_ne!(sig1, sig3);
    }

    #[test]
    fn test_reduce_lookup_found() {
        let reg = init_default_loops();
        let result = reg.lookup_reduce(ReduceOp::Sum, DTypeKind::Float64);
        assert!(result.is_some());
        let (_, _, out_dtype) = result.unwrap();
        assert_eq!(out_dtype, DTypeKind::Float64);
    }

    #[test]
    fn test_f64_sum_loop() {
        let reg = init_default_loops();
        let (init_fn, acc_fn, _) = reg
            .lookup_reduce(ReduceOp::Sum, DTypeKind::Float64)
            .unwrap();

        let values: [f64; 3] = [1.0, 2.0, 3.0];
        let mut acc: f64 = 0.0;

        unsafe {
            init_fn(&mut acc as *mut f64 as *mut u8, 0);
            for v in &values {
                acc_fn(
                    &mut acc as *mut f64 as *mut u8,
                    0,
                    v as *const f64 as *const u8,
                    0,
                );
            }
        }

        assert_eq!(acc, 6.0);
    }

    #[test]
    fn test_f64_max_loop() {
        let reg = init_default_loops();
        let (init_fn, acc_fn, _) = reg
            .lookup_reduce(ReduceOp::Max, DTypeKind::Float64)
            .unwrap();

        let values: [f64; 4] = [1.0, 5.0, 2.0, 3.0];
        let mut acc: f64 = 0.0;

        unsafe {
            init_fn(&mut acc as *mut f64 as *mut u8, 0);
            for v in &values {
                acc_fn(
                    &mut acc as *mut f64 as *mut u8,
                    0,
                    v as *const f64 as *const u8,
                    0,
                );
            }
        }

        assert_eq!(acc, 5.0);
    }

    #[test]
    fn test_i64_prod_loop() {
        let reg = init_default_loops();
        let (init_fn, acc_fn, _) = reg
            .lookup_reduce(ReduceOp::Prod, DTypeKind::Int64)
            .unwrap();

        let values: [i64; 4] = [2, 3, 4, 5];
        let mut acc: i64 = 0;

        unsafe {
            init_fn(&mut acc as *mut i64 as *mut u8, 0);
            for v in &values {
                acc_fn(
                    &mut acc as *mut i64 as *mut u8,
                    0,
                    v as *const i64 as *const u8,
                    0,
                );
            }
        }

        assert_eq!(acc, 120);
    }
}
