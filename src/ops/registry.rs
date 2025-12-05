//! UFunc registry for type-specific inner loops.
//!
//! Provides NumPy-style dispatch: registered loops checked first,
//! then fallback to DTypeOps trait methods.

use crate::array::dtype::{BinaryOp, DTypeKind, ReduceOp, UnaryOp};
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

/// Type signature for a ufunc loop.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TypeSignature {
    pub inputs: Vec<DTypeKind>,
    pub output: DTypeKind,
}

impl TypeSignature {
    pub fn binary(a: DTypeKind, b: DTypeKind, out: DTypeKind) -> Self {
        Self {
            inputs: vec![a, b],
            output: out,
        }
    }

    pub fn unary(input: DTypeKind, out: DTypeKind) -> Self {
        Self {
            inputs: vec![input],
            output: out,
        }
    }

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

/// Accumulate one element into reduction.
pub type ReduceAccFn = unsafe fn(
    acc_ptr: *mut u8,
    idx: usize,
    val_ptr: *const u8,
    byte_offset: isize,
);

/// Registry for ufunc inner loops.
pub struct UFuncRegistry {
    binary_loops: HashMap<(BinaryOp, TypeSignature), BinaryLoopFn>,
    unary_loops: HashMap<(UnaryOp, TypeSignature), UnaryLoopFn>,
    reduce_loops: HashMap<(ReduceOp, TypeSignature), (ReduceInitFn, ReduceAccFn)>,
}

impl UFuncRegistry {
    pub fn new() -> Self {
        Self {
            binary_loops: HashMap::new(),
            unary_loops: HashMap::new(),
            reduce_loops: HashMap::new(),
        }
    }

    /// Register a binary loop for a specific operation and type signature.
    pub fn register_binary(&mut self, op: BinaryOp, sig: TypeSignature, f: BinaryLoopFn) {
        self.binary_loops.insert((op, sig), f);
    }

    /// Register a unary loop for a specific operation and type signature.
    pub fn register_unary(&mut self, op: UnaryOp, sig: TypeSignature, f: UnaryLoopFn) {
        self.unary_loops.insert((op, sig), f);
    }

    /// Look up a binary loop. Returns None if no registered loop exists.
    pub fn lookup_binary(
        &self,
        op: BinaryOp,
        a: DTypeKind,
        b: DTypeKind,
    ) -> Option<(BinaryLoopFn, DTypeKind)> {
        // For now, only check exact same-type matches (a == b == out)
        let sig = TypeSignature::binary(a.clone(), b.clone(), a.clone());
        if a == b {
            self.binary_loops.get(&(op, sig)).map(|&f| (f, a))
        } else {
            None
        }
    }

    /// Look up a unary loop. Returns None if no registered loop exists.
    pub fn lookup_unary(
        &self,
        op: UnaryOp,
        input: DTypeKind,
    ) -> Option<(UnaryLoopFn, DTypeKind)> {
        let sig = TypeSignature::unary(input.clone(), input.clone());
        self.unary_loops.get(&(op, sig)).map(|&f| (f, input))
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
fn init_default_loops() -> UFuncRegistry {
    let mut reg = UFuncRegistry::new();

    // Macro for strided binary loops with contiguous fast path
    macro_rules! register_strided_binary {
        ($reg:expr, $op:expr, $kind:expr, $T:ty, $f:expr) => {
            $reg.register_binary(
                $op,
                TypeSignature::binary($kind.clone(), $kind.clone(), $kind.clone()),
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

    // f64 binary loops
    register_strided_binary!(reg, BinaryOp::Add, DTypeKind::Float64, f64, |a, b| a + b);
    register_strided_binary!(reg, BinaryOp::Sub, DTypeKind::Float64, f64, |a, b| a - b);
    register_strided_binary!(reg, BinaryOp::Mul, DTypeKind::Float64, f64, |a, b| a * b);
    register_strided_binary!(reg, BinaryOp::Div, DTypeKind::Float64, f64, |a: f64, b: f64| a / b);

    // f32 binary loops
    register_strided_binary!(reg, BinaryOp::Add, DTypeKind::Float32, f32, |a, b| a + b);
    register_strided_binary!(reg, BinaryOp::Sub, DTypeKind::Float32, f32, |a, b| a - b);
    register_strided_binary!(reg, BinaryOp::Mul, DTypeKind::Float32, f32, |a, b| a * b);
    register_strided_binary!(reg, BinaryOp::Div, DTypeKind::Float32, f32, |a: f32, b: f32| a / b);

    // i64 binary loops
    register_strided_binary!(reg, BinaryOp::Add, DTypeKind::Int64, i64, |a, b| a + b);
    register_strided_binary!(reg, BinaryOp::Sub, DTypeKind::Int64, i64, |a, b| a - b);
    register_strided_binary!(reg, BinaryOp::Mul, DTypeKind::Int64, i64, |a, b| a * b);
    register_strided_binary!(reg, BinaryOp::Div, DTypeKind::Int64, i64, |a, b| a / b);

    // i32 binary loops
    register_strided_binary!(reg, BinaryOp::Add, DTypeKind::Int32, i32, |a, b| a + b);
    register_strided_binary!(reg, BinaryOp::Sub, DTypeKind::Int32, i32, |a, b| a - b);
    register_strided_binary!(reg, BinaryOp::Mul, DTypeKind::Int32, i32, |a, b| a * b);
    register_strided_binary!(reg, BinaryOp::Div, DTypeKind::Int32, i32, |a, b| a / b);

    // u64 binary loops
    register_strided_binary!(reg, BinaryOp::Add, DTypeKind::Uint64, u64, |a, b| a + b);
    register_strided_binary!(reg, BinaryOp::Sub, DTypeKind::Uint64, u64, |a, b| a - b);
    register_strided_binary!(reg, BinaryOp::Mul, DTypeKind::Uint64, u64, |a, b| a * b);
    register_strided_binary!(reg, BinaryOp::Div, DTypeKind::Uint64, u64, |a, b| a / b);

    // u32 binary loops
    register_strided_binary!(reg, BinaryOp::Add, DTypeKind::Uint32, u32, |a, b| a + b);
    register_strided_binary!(reg, BinaryOp::Sub, DTypeKind::Uint32, u32, |a, b| a - b);
    register_strided_binary!(reg, BinaryOp::Mul, DTypeKind::Uint32, u32, |a, b| a * b);
    register_strided_binary!(reg, BinaryOp::Div, DTypeKind::Uint32, u32, |a, b| a / b);

    // u8 binary loops
    register_strided_binary!(reg, BinaryOp::Add, DTypeKind::Uint8, u8, |a, b| a + b);
    register_strided_binary!(reg, BinaryOp::Sub, DTypeKind::Uint8, u8, |a, b| a - b);
    register_strided_binary!(reg, BinaryOp::Mul, DTypeKind::Uint8, u8, |a, b| a * b);
    register_strided_binary!(reg, BinaryOp::Div, DTypeKind::Uint8, u8, |a, b| a / b);

    // Bool binary loops (Add=or, Mul=and)
    reg.register_binary(
        BinaryOp::Add,
        TypeSignature::binary(DTypeKind::Bool, DTypeKind::Bool, DTypeKind::Bool),
        |a_ptr, b_ptr, out_ptr, n, strides| unsafe {
            let (sa, sb, so) = strides;
            let mut ap = a_ptr;
            let mut bp = b_ptr;
            let mut op = out_ptr;
            for _ in 0..n {
                let a = *(ap as *const u8) != 0;
                let b = *(bp as *const u8) != 0;
                *(op as *mut u8) = (a || b) as u8;
                ap = ap.offset(sa);
                bp = bp.offset(sb);
                op = op.offset(so);
            }
        },
    );
    reg.register_binary(
        BinaryOp::Mul,
        TypeSignature::binary(DTypeKind::Bool, DTypeKind::Bool, DTypeKind::Bool),
        |a_ptr, b_ptr, out_ptr, n, strides| unsafe {
            let (sa, sb, so) = strides;
            let mut ap = a_ptr;
            let mut bp = b_ptr;
            let mut op = out_ptr;
            for _ in 0..n {
                let a = *(ap as *const u8) != 0;
                let b = *(bp as *const u8) != 0;
                *(op as *mut u8) = (a && b) as u8;
                ap = ap.offset(sa);
                bp = bp.offset(sb);
                op = op.offset(so);
            }
        },
    );

    // Complex128 binary loops
    reg.register_binary(
        BinaryOp::Add,
        TypeSignature::binary(DTypeKind::Complex128, DTypeKind::Complex128, DTypeKind::Complex128),
        |a_ptr, b_ptr, out_ptr, n, strides| unsafe {
            let (sa, sb, so) = strides;
            let mut ap = a_ptr;
            let mut bp = b_ptr;
            let mut op = out_ptr;
            for _ in 0..n {
                let a = ap as *const f64;
                let b = bp as *const f64;
                let out = op as *mut f64;
                *out = *a + *b;
                *out.add(1) = *a.add(1) + *b.add(1);
                ap = ap.offset(sa);
                bp = bp.offset(sb);
                op = op.offset(so);
            }
        },
    );
    reg.register_binary(
        BinaryOp::Sub,
        TypeSignature::binary(DTypeKind::Complex128, DTypeKind::Complex128, DTypeKind::Complex128),
        |a_ptr, b_ptr, out_ptr, n, strides| unsafe {
            let (sa, sb, so) = strides;
            let mut ap = a_ptr;
            let mut bp = b_ptr;
            let mut op = out_ptr;
            for _ in 0..n {
                let a = ap as *const f64;
                let b = bp as *const f64;
                let out = op as *mut f64;
                *out = *a - *b;
                *out.add(1) = *a.add(1) - *b.add(1);
                ap = ap.offset(sa);
                bp = bp.offset(sb);
                op = op.offset(so);
            }
        },
    );
    reg.register_binary(
        BinaryOp::Mul,
        TypeSignature::binary(DTypeKind::Complex128, DTypeKind::Complex128, DTypeKind::Complex128),
        |a_ptr, b_ptr, out_ptr, n, strides| unsafe {
            let (sa, sb, so) = strides;
            let mut ap = a_ptr;
            let mut bp = b_ptr;
            let mut op = out_ptr;
            for _ in 0..n {
                let a = ap as *const f64;
                let b = bp as *const f64;
                let ar = *a; let ai = *a.add(1);
                let br = *b; let bi = *b.add(1);
                let out = op as *mut f64;
                *out = ar * br - ai * bi;
                *out.add(1) = ar * bi + ai * br;
                ap = ap.offset(sa);
                bp = bp.offset(sb);
                op = op.offset(so);
            }
        },
    );
    reg.register_binary(
        BinaryOp::Div,
        TypeSignature::binary(DTypeKind::Complex128, DTypeKind::Complex128, DTypeKind::Complex128),
        |a_ptr, b_ptr, out_ptr, n, strides| unsafe {
            let (sa, sb, so) = strides;
            let mut ap = a_ptr;
            let mut bp = b_ptr;
            let mut op = out_ptr;
            for _ in 0..n {
                let a = ap as *const f64;
                let b = bp as *const f64;
                let ar = *a; let ai = *a.add(1);
                let br = *b; let bi = *b.add(1);
                let denom = br * br + bi * bi;
                let out = op as *mut f64;
                if denom != 0.0 {
                    *out = (ar * br + ai * bi) / denom;
                    *out.add(1) = (ai * br - ar * bi) / denom;
                } else {
                    *out = f64::NAN;
                    *out.add(1) = f64::NAN;
                }
                ap = ap.offset(sa);
                bp = bp.offset(sb);
                op = op.offset(so);
            }
        },
    );

    // ========================================================================
    // Unary loops
    // ========================================================================

    // Macro for strided unary loops with contiguous fast path
    macro_rules! register_strided_unary {
        ($reg:expr, $op:expr, $kind:expr, $T:ty, $f:expr) => {
            $reg.register_unary(
                $op,
                TypeSignature::unary($kind.clone(), $kind.clone()),
                |src_ptr, out_ptr, n, strides| unsafe {
                    let itemsize = std::mem::size_of::<$T>() as isize;
                    let (ss, so) = strides;
                    if ss == itemsize && so == itemsize {
                        // Contiguous fast path - LLVM can auto-vectorize
                        let src = std::slice::from_raw_parts(src_ptr as *const $T, n);
                        let out = std::slice::from_raw_parts_mut(out_ptr as *mut $T, n);
                        for i in 0..n {
                            out[i] = $f(src[i]);
                        }
                    } else {
                        // Strided path
                        let mut sp = src_ptr;
                        let mut op = out_ptr;
                        for _ in 0..n {
                            let v = *(sp as *const $T);
                            *(op as *mut $T) = $f(v);
                            sp = sp.offset(ss);
                            op = op.offset(so);
                        }
                    }
                },
            );
        };
    }

    // f64 unary loops
    register_strided_unary!(reg, UnaryOp::Neg, DTypeKind::Float64, f64, |v: f64| -v);
    register_strided_unary!(reg, UnaryOp::Abs, DTypeKind::Float64, f64, |v: f64| v.abs());
    register_strided_unary!(reg, UnaryOp::Sqrt, DTypeKind::Float64, f64, |v: f64| v.sqrt());
    register_strided_unary!(reg, UnaryOp::Exp, DTypeKind::Float64, f64, |v: f64| v.exp());
    register_strided_unary!(reg, UnaryOp::Log, DTypeKind::Float64, f64, |v: f64| v.ln());
    register_strided_unary!(reg, UnaryOp::Sin, DTypeKind::Float64, f64, |v: f64| v.sin());
    register_strided_unary!(reg, UnaryOp::Cos, DTypeKind::Float64, f64, |v: f64| v.cos());
    register_strided_unary!(reg, UnaryOp::Tan, DTypeKind::Float64, f64, |v: f64| v.tan());
    register_strided_unary!(reg, UnaryOp::Floor, DTypeKind::Float64, f64, |v: f64| v.floor());
    register_strided_unary!(reg, UnaryOp::Ceil, DTypeKind::Float64, f64, |v: f64| v.ceil());
    register_strided_unary!(reg, UnaryOp::Arcsin, DTypeKind::Float64, f64, |v: f64| v.asin());
    register_strided_unary!(reg, UnaryOp::Arccos, DTypeKind::Float64, f64, |v: f64| v.acos());
    register_strided_unary!(reg, UnaryOp::Arctan, DTypeKind::Float64, f64, |v: f64| v.atan());

    // f32 unary loops
    register_strided_unary!(reg, UnaryOp::Neg, DTypeKind::Float32, f32, |v: f32| -v);
    register_strided_unary!(reg, UnaryOp::Abs, DTypeKind::Float32, f32, |v: f32| v.abs());
    register_strided_unary!(reg, UnaryOp::Sqrt, DTypeKind::Float32, f32, |v: f32| v.sqrt());
    register_strided_unary!(reg, UnaryOp::Floor, DTypeKind::Float32, f32, |v: f32| v.floor());
    register_strided_unary!(reg, UnaryOp::Ceil, DTypeKind::Float32, f32, |v: f32| v.ceil());
    register_strided_unary!(reg, UnaryOp::Arcsin, DTypeKind::Float32, f32, |v: f32| v.asin());
    register_strided_unary!(reg, UnaryOp::Arccos, DTypeKind::Float32, f32, |v: f32| v.acos());
    register_strided_unary!(reg, UnaryOp::Arctan, DTypeKind::Float32, f32, |v: f32| v.atan());

    // i64 unary loops
    register_strided_unary!(reg, UnaryOp::Neg, DTypeKind::Int64, i64, |v: i64| -v);
    register_strided_unary!(reg, UnaryOp::Abs, DTypeKind::Int64, i64, |v: i64| v.abs());

    // i32 unary loops
    register_strided_unary!(reg, UnaryOp::Neg, DTypeKind::Int32, i32, |v: i32| -v);
    register_strided_unary!(reg, UnaryOp::Abs, DTypeKind::Int32, i32, |v: i32| v.abs());

    // uint64 unary loops (Abs only - no Neg for unsigned)
    register_strided_unary!(reg, UnaryOp::Abs, DTypeKind::Uint64, u64, |v: u64| v);

    // uint32 unary loops
    register_strided_unary!(reg, UnaryOp::Abs, DTypeKind::Uint32, u32, |v: u32| v);

    // uint8 unary loops
    register_strided_unary!(reg, UnaryOp::Abs, DTypeKind::Uint8, u8, |v: u8| v);

    // complex128 unary loops (no contiguous fast path - needs special handling)
    reg.register_unary(
        UnaryOp::Neg,
        TypeSignature::unary(DTypeKind::Complex128, DTypeKind::Complex128),
        |src_ptr, out_ptr, n, strides| unsafe {
            let (ss, so) = strides;
            let mut sp = src_ptr;
            let mut op = out_ptr;
            for _ in 0..n {
                let src = sp as *const f64;
                let out = op as *mut f64;
                *out = -(*src);
                *out.add(1) = -(*src.add(1));
                sp = sp.offset(ss);
                op = op.offset(so);
            }
        },
    );

    // ========================================================================
    // Reduce loops
    // ========================================================================

    // f64 reduce loops
    reg.register_reduce(
        ReduceOp::Sum,
        TypeSignature::reduce(DTypeKind::Float64, DTypeKind::Float64),
        |out_ptr, idx| unsafe { *(out_ptr as *mut f64).add(idx) = 0.0; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut f64).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const f64);
            *acc += v;
        },
    );
    reg.register_reduce(
        ReduceOp::Prod,
        TypeSignature::reduce(DTypeKind::Float64, DTypeKind::Float64),
        |out_ptr, idx| unsafe { *(out_ptr as *mut f64).add(idx) = 1.0; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut f64).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const f64);
            *acc *= v;
        },
    );
    reg.register_reduce(
        ReduceOp::Max,
        TypeSignature::reduce(DTypeKind::Float64, DTypeKind::Float64),
        |out_ptr, idx| unsafe { *(out_ptr as *mut f64).add(idx) = f64::NEG_INFINITY; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut f64).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const f64);
            if v > *acc { *acc = v; }
        },
    );
    reg.register_reduce(
        ReduceOp::Min,
        TypeSignature::reduce(DTypeKind::Float64, DTypeKind::Float64),
        |out_ptr, idx| unsafe { *(out_ptr as *mut f64).add(idx) = f64::INFINITY; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut f64).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const f64);
            if v < *acc { *acc = v; }
        },
    );

    // f32 reduce loops
    reg.register_reduce(
        ReduceOp::Sum,
        TypeSignature::reduce(DTypeKind::Float32, DTypeKind::Float32),
        |out_ptr, idx| unsafe { *(out_ptr as *mut f32).add(idx) = 0.0; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut f32).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const f32);
            *acc += v;
        },
    );
    reg.register_reduce(
        ReduceOp::Prod,
        TypeSignature::reduce(DTypeKind::Float32, DTypeKind::Float32),
        |out_ptr, idx| unsafe { *(out_ptr as *mut f32).add(idx) = 1.0; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut f32).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const f32);
            *acc *= v;
        },
    );
    reg.register_reduce(
        ReduceOp::Max,
        TypeSignature::reduce(DTypeKind::Float32, DTypeKind::Float32),
        |out_ptr, idx| unsafe { *(out_ptr as *mut f32).add(idx) = f32::NEG_INFINITY; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut f32).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const f32);
            if v > *acc { *acc = v; }
        },
    );
    reg.register_reduce(
        ReduceOp::Min,
        TypeSignature::reduce(DTypeKind::Float32, DTypeKind::Float32),
        |out_ptr, idx| unsafe { *(out_ptr as *mut f32).add(idx) = f32::INFINITY; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut f32).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const f32);
            if v < *acc { *acc = v; }
        },
    );

    // i64 reduce loops
    reg.register_reduce(
        ReduceOp::Sum,
        TypeSignature::reduce(DTypeKind::Int64, DTypeKind::Int64),
        |out_ptr, idx| unsafe { *(out_ptr as *mut i64).add(idx) = 0; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut i64).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const i64);
            *acc = acc.read().wrapping_add(v);
        },
    );
    reg.register_reduce(
        ReduceOp::Prod,
        TypeSignature::reduce(DTypeKind::Int64, DTypeKind::Int64),
        |out_ptr, idx| unsafe { *(out_ptr as *mut i64).add(idx) = 1; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut i64).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const i64);
            *acc = acc.read().wrapping_mul(v);
        },
    );
    reg.register_reduce(
        ReduceOp::Max,
        TypeSignature::reduce(DTypeKind::Int64, DTypeKind::Int64),
        |out_ptr, idx| unsafe { *(out_ptr as *mut i64).add(idx) = i64::MIN; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut i64).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const i64);
            if v > *acc { *acc = v; }
        },
    );
    reg.register_reduce(
        ReduceOp::Min,
        TypeSignature::reduce(DTypeKind::Int64, DTypeKind::Int64),
        |out_ptr, idx| unsafe { *(out_ptr as *mut i64).add(idx) = i64::MAX; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut i64).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const i64);
            if v < *acc { *acc = v; }
        },
    );

    // i32 reduce loops
    reg.register_reduce(
        ReduceOp::Sum,
        TypeSignature::reduce(DTypeKind::Int32, DTypeKind::Int32),
        |out_ptr, idx| unsafe { *(out_ptr as *mut i32).add(idx) = 0; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut i32).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const i32);
            *acc = acc.read().wrapping_add(v);
        },
    );
    reg.register_reduce(
        ReduceOp::Prod,
        TypeSignature::reduce(DTypeKind::Int32, DTypeKind::Int32),
        |out_ptr, idx| unsafe { *(out_ptr as *mut i32).add(idx) = 1; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut i32).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const i32);
            *acc = acc.read().wrapping_mul(v);
        },
    );
    reg.register_reduce(
        ReduceOp::Max,
        TypeSignature::reduce(DTypeKind::Int32, DTypeKind::Int32),
        |out_ptr, idx| unsafe { *(out_ptr as *mut i32).add(idx) = i32::MIN; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut i32).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const i32);
            if v > *acc { *acc = v; }
        },
    );
    reg.register_reduce(
        ReduceOp::Min,
        TypeSignature::reduce(DTypeKind::Int32, DTypeKind::Int32),
        |out_ptr, idx| unsafe { *(out_ptr as *mut i32).add(idx) = i32::MAX; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut i32).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const i32);
            if v < *acc { *acc = v; }
        },
    );

    // uint64 reduce loops
    reg.register_reduce(
        ReduceOp::Sum,
        TypeSignature::reduce(DTypeKind::Uint64, DTypeKind::Uint64),
        |out_ptr, idx| unsafe { *(out_ptr as *mut u64).add(idx) = 0; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut u64).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const u64);
            *acc = acc.read().wrapping_add(v);
        },
    );
    reg.register_reduce(
        ReduceOp::Prod,
        TypeSignature::reduce(DTypeKind::Uint64, DTypeKind::Uint64),
        |out_ptr, idx| unsafe { *(out_ptr as *mut u64).add(idx) = 1; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut u64).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const u64);
            *acc = acc.read().wrapping_mul(v);
        },
    );
    reg.register_reduce(
        ReduceOp::Max,
        TypeSignature::reduce(DTypeKind::Uint64, DTypeKind::Uint64),
        |out_ptr, idx| unsafe { *(out_ptr as *mut u64).add(idx) = u64::MIN; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut u64).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const u64);
            if v > *acc { *acc = v; }
        },
    );
    reg.register_reduce(
        ReduceOp::Min,
        TypeSignature::reduce(DTypeKind::Uint64, DTypeKind::Uint64),
        |out_ptr, idx| unsafe { *(out_ptr as *mut u64).add(idx) = u64::MAX; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut u64).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const u64);
            if v < *acc { *acc = v; }
        },
    );

    // uint32 reduce loops
    reg.register_reduce(
        ReduceOp::Sum,
        TypeSignature::reduce(DTypeKind::Uint32, DTypeKind::Uint32),
        |out_ptr, idx| unsafe { *(out_ptr as *mut u32).add(idx) = 0; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut u32).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const u32);
            *acc = acc.read().wrapping_add(v);
        },
    );
    reg.register_reduce(
        ReduceOp::Prod,
        TypeSignature::reduce(DTypeKind::Uint32, DTypeKind::Uint32),
        |out_ptr, idx| unsafe { *(out_ptr as *mut u32).add(idx) = 1; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut u32).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const u32);
            *acc = acc.read().wrapping_mul(v);
        },
    );
    reg.register_reduce(
        ReduceOp::Max,
        TypeSignature::reduce(DTypeKind::Uint32, DTypeKind::Uint32),
        |out_ptr, idx| unsafe { *(out_ptr as *mut u32).add(idx) = u32::MIN; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut u32).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const u32);
            if v > *acc { *acc = v; }
        },
    );
    reg.register_reduce(
        ReduceOp::Min,
        TypeSignature::reduce(DTypeKind::Uint32, DTypeKind::Uint32),
        |out_ptr, idx| unsafe { *(out_ptr as *mut u32).add(idx) = u32::MAX; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut u32).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const u32);
            if v < *acc { *acc = v; }
        },
    );

    // uint8 reduce loops
    reg.register_reduce(
        ReduceOp::Sum,
        TypeSignature::reduce(DTypeKind::Uint8, DTypeKind::Uint8),
        |out_ptr, idx| unsafe { *(out_ptr as *mut u8).add(idx) = 0; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut u8).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const u8);
            *acc = acc.read().wrapping_add(v);
        },
    );
    reg.register_reduce(
        ReduceOp::Prod,
        TypeSignature::reduce(DTypeKind::Uint8, DTypeKind::Uint8),
        |out_ptr, idx| unsafe { *(out_ptr as *mut u8).add(idx) = 1; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut u8).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const u8);
            *acc = acc.read().wrapping_mul(v);
        },
    );
    reg.register_reduce(
        ReduceOp::Max,
        TypeSignature::reduce(DTypeKind::Uint8, DTypeKind::Uint8),
        |out_ptr, idx| unsafe { *(out_ptr as *mut u8).add(idx) = u8::MIN; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut u8).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const u8);
            if v > *acc { *acc = v; }
        },
    );
    reg.register_reduce(
        ReduceOp::Min,
        TypeSignature::reduce(DTypeKind::Uint8, DTypeKind::Uint8),
        |out_ptr, idx| unsafe { *(out_ptr as *mut u8).add(idx) = u8::MAX; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut u8).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const u8);
            if v < *acc { *acc = v; }
        },
    );

    // bool reduce loops (Sum=any, Prod=all)
    reg.register_reduce(
        ReduceOp::Sum,
        TypeSignature::reduce(DTypeKind::Bool, DTypeKind::Bool),
        |out_ptr, idx| unsafe { *(out_ptr as *mut u8).add(idx) = 0; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut u8).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const u8);
            if v != 0 { *acc = 1; }
        },
    );
    reg.register_reduce(
        ReduceOp::Prod,
        TypeSignature::reduce(DTypeKind::Bool, DTypeKind::Bool),
        |out_ptr, idx| unsafe { *(out_ptr as *mut u8).add(idx) = 1; },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut u8).add(idx);
            let v = *(val_ptr.offset(byte_offset) as *const u8);
            if v == 0 { *acc = 0; }
        },
    );

    // complex128 reduce loops (Sum, Prod only - Max/Min not well-defined)
    reg.register_reduce(
        ReduceOp::Sum,
        TypeSignature::reduce(DTypeKind::Complex128, DTypeKind::Complex128),
        |out_ptr, idx| unsafe {
            let out = (out_ptr as *mut f64).add(idx * 2);
            *out = 0.0;
            *out.add(1) = 0.0;
        },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut f64).add(idx * 2);
            let v = val_ptr.offset(byte_offset) as *const f64;
            *acc += *v;
            *acc.add(1) += *v.add(1);
        },
    );
    reg.register_reduce(
        ReduceOp::Prod,
        TypeSignature::reduce(DTypeKind::Complex128, DTypeKind::Complex128),
        |out_ptr, idx| unsafe {
            let out = (out_ptr as *mut f64).add(idx * 2);
            *out = 1.0;
            *out.add(1) = 0.0;
        },
        |acc_ptr, idx, val_ptr, byte_offset| unsafe {
            let acc = (acc_ptr as *mut f64).add(idx * 2);
            let v = val_ptr.offset(byte_offset) as *const f64;
            let ar = *acc; let ai = *acc.add(1);
            let vr = *v; let vi = *v.add(1);
            *acc = ar * vr - ai * vi;
            *acc.add(1) = ar * vi + ai * vr;
        },
    );

    reg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_signature_equality() {
        let sig1 = TypeSignature::binary(DTypeKind::Float64, DTypeKind::Float64, DTypeKind::Float64);
        let sig2 = TypeSignature::binary(DTypeKind::Float64, DTypeKind::Float64, DTypeKind::Float64);
        let sig3 = TypeSignature::binary(DTypeKind::Float32, DTypeKind::Float32, DTypeKind::Float32);

        assert_eq!(sig1, sig2);
        assert_ne!(sig1, sig3);
    }

    #[test]
    fn test_registry_lookup_found() {
        let reg = init_default_loops();
        let result = reg.lookup_binary(BinaryOp::Add, DTypeKind::Float64, DTypeKind::Float64);
        assert!(result.is_some());
        let (_, out_dtype) = result.unwrap();
        assert_eq!(out_dtype, DTypeKind::Float64);
    }

    #[test]
    fn test_registry_lookup_not_found() {
        let reg = init_default_loops();
        // Mixed types not registered
        let result = reg.lookup_binary(BinaryOp::Add, DTypeKind::Float64, DTypeKind::Float32);
        assert!(result.is_none());
    }

    #[test]
    fn test_f64_add_loop() {
        let reg = init_default_loops();
        let (loop_fn, _) = reg
            .lookup_binary(BinaryOp::Add, DTypeKind::Float64, DTypeKind::Float64)
            .unwrap();

        let a: [f64; 2] = [3.0, 10.0];
        let b: [f64; 2] = [4.0, 20.0];
        let mut out: [f64; 2] = [0.0, 0.0];

        unsafe {
            loop_fn(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                out.as_mut_ptr() as *mut u8,
                2,
                (8, 8, 8),  // f64 strides
            );
        }

        assert_eq!(out, [7.0, 30.0]);
    }

    #[test]
    fn test_i64_mul_loop() {
        let reg = init_default_loops();
        let (loop_fn, _) = reg
            .lookup_binary(BinaryOp::Mul, DTypeKind::Int64, DTypeKind::Int64)
            .unwrap();

        let a: [i64; 2] = [5, 7];
        let b: [i64; 2] = [6, 8];
        let mut out: [i64; 2] = [0, 0];

        unsafe {
            loop_fn(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                out.as_mut_ptr() as *mut u8,
                2,
                (8, 8, 8),  // i64 strides
            );
        }

        assert_eq!(out, [30, 56]);
    }

    #[test]
    fn test_global_registry() {
        let reg = registry().read().unwrap();
        let result = reg.lookup_binary(BinaryOp::Add, DTypeKind::Float64, DTypeKind::Float64);
        assert!(result.is_some());
    }

    #[test]
    fn test_f64_neg_loop() {
        let reg = init_default_loops();
        let (loop_fn, _) = reg
            .lookup_unary(UnaryOp::Neg, DTypeKind::Float64)
            .unwrap();

        let src: [f64; 2] = [3.5, -2.0];
        let mut out: [f64; 2] = [0.0, 0.0];

        unsafe {
            loop_fn(
                src.as_ptr() as *const u8,
                out.as_mut_ptr() as *mut u8,
                2,
                (8, 8),  // f64 strides
            );
        }

        assert_eq!(out, [-3.5, 2.0]);
    }

    #[test]
    fn test_f64_sqrt_loop() {
        let reg = init_default_loops();
        let (loop_fn, _) = reg
            .lookup_unary(UnaryOp::Sqrt, DTypeKind::Float64)
            .unwrap();

        let src: [f64; 2] = [16.0, 25.0];
        let mut out: [f64; 2] = [0.0, 0.0];

        unsafe {
            loop_fn(
                src.as_ptr() as *const u8,
                out.as_mut_ptr() as *mut u8,
                2,
                (8, 8),  // f64 strides
            );
        }

        assert_eq!(out, [4.0, 5.0]);
    }

    #[test]
    fn test_unary_lookup_not_found() {
        let reg = init_default_loops();
        // Exp not registered for i32
        let result = reg.lookup_unary(UnaryOp::Exp, DTypeKind::Int32);
        assert!(result.is_none());
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
