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
/// # Safety
/// Caller must ensure pointers are valid and offsets are within bounds.
pub type BinaryLoopFn = unsafe fn(
    a_ptr: *const u8,
    a_offset: isize,
    b_ptr: *const u8,
    b_offset: isize,
    out_ptr: *mut u8,
    out_idx: usize,
);

/// Inner loop function for unary operations.
pub type UnaryLoopFn = unsafe fn(
    src_ptr: *const u8,
    src_offset: isize,
    out_ptr: *mut u8,
    out_idx: usize,
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

    // Register f64 binary loops
    macro_rules! register_f64_binary {
        ($op:expr, $rust_op:tt) => {
            reg.register_binary(
                $op,
                TypeSignature::binary(DTypeKind::Float64, DTypeKind::Float64, DTypeKind::Float64),
                |a_ptr, a_off, b_ptr, b_off, out_ptr, out_idx| unsafe {
                    let a = *(a_ptr.offset(a_off) as *const f64);
                    let b = *(b_ptr.offset(b_off) as *const f64);
                    let out = out_ptr as *mut f64;
                    *out.add(out_idx) = a $rust_op b;
                },
            );
        };
    }

    register_f64_binary!(BinaryOp::Add, +);
    register_f64_binary!(BinaryOp::Sub, -);
    register_f64_binary!(BinaryOp::Mul, *);

    // Division needs special handling for zero
    reg.register_binary(
        BinaryOp::Div,
        TypeSignature::binary(DTypeKind::Float64, DTypeKind::Float64, DTypeKind::Float64),
        |a_ptr, a_off, b_ptr, b_off, out_ptr, out_idx| unsafe {
            let a = *(a_ptr.offset(a_off) as *const f64);
            let b = *(b_ptr.offset(b_off) as *const f64);
            let out = out_ptr as *mut f64;
            *out.add(out_idx) = if b != 0.0 { a / b } else { f64::NAN };
        },
    );

    // Register f32 binary loops
    macro_rules! register_f32_binary {
        ($op:expr, $rust_op:tt) => {
            reg.register_binary(
                $op,
                TypeSignature::binary(DTypeKind::Float32, DTypeKind::Float32, DTypeKind::Float32),
                |a_ptr, a_off, b_ptr, b_off, out_ptr, out_idx| unsafe {
                    let a = *(a_ptr.offset(a_off) as *const f32);
                    let b = *(b_ptr.offset(b_off) as *const f32);
                    let out = out_ptr as *mut f32;
                    *out.add(out_idx) = a $rust_op b;
                },
            );
        };
    }

    register_f32_binary!(BinaryOp::Add, +);
    register_f32_binary!(BinaryOp::Sub, -);
    register_f32_binary!(BinaryOp::Mul, *);

    reg.register_binary(
        BinaryOp::Div,
        TypeSignature::binary(DTypeKind::Float32, DTypeKind::Float32, DTypeKind::Float32),
        |a_ptr, a_off, b_ptr, b_off, out_ptr, out_idx| unsafe {
            let a = *(a_ptr.offset(a_off) as *const f32);
            let b = *(b_ptr.offset(b_off) as *const f32);
            let out = out_ptr as *mut f32;
            *out.add(out_idx) = if b != 0.0 { a / b } else { f32::NAN };
        },
    );

    // Register i64 binary loops
    macro_rules! register_i64_binary {
        ($op:expr, $rust_op:tt) => {
            reg.register_binary(
                $op,
                TypeSignature::binary(DTypeKind::Int64, DTypeKind::Int64, DTypeKind::Int64),
                |a_ptr, a_off, b_ptr, b_off, out_ptr, out_idx| unsafe {
                    let a = *(a_ptr.offset(a_off) as *const i64);
                    let b = *(b_ptr.offset(b_off) as *const i64);
                    let out = out_ptr as *mut i64;
                    *out.add(out_idx) = a $rust_op b;
                },
            );
        };
    }

    register_i64_binary!(BinaryOp::Add, +);
    register_i64_binary!(BinaryOp::Sub, -);
    register_i64_binary!(BinaryOp::Mul, *);
    register_i64_binary!(BinaryOp::Div, /);

    // Register i32 binary loops
    macro_rules! register_i32_binary {
        ($op:expr, $rust_op:tt) => {
            reg.register_binary(
                $op,
                TypeSignature::binary(DTypeKind::Int32, DTypeKind::Int32, DTypeKind::Int32),
                |a_ptr, a_off, b_ptr, b_off, out_ptr, out_idx| unsafe {
                    let a = *(a_ptr.offset(a_off) as *const i32);
                    let b = *(b_ptr.offset(b_off) as *const i32);
                    let out = out_ptr as *mut i32;
                    *out.add(out_idx) = a $rust_op b;
                },
            );
        };
    }

    register_i32_binary!(BinaryOp::Add, +);
    register_i32_binary!(BinaryOp::Sub, -);
    register_i32_binary!(BinaryOp::Mul, *);
    register_i32_binary!(BinaryOp::Div, /);

    // Register uint64 binary loops
    macro_rules! register_u64_binary {
        ($op:expr, $rust_op:tt) => {
            reg.register_binary(
                $op,
                TypeSignature::binary(DTypeKind::Uint64, DTypeKind::Uint64, DTypeKind::Uint64),
                |a_ptr, a_off, b_ptr, b_off, out_ptr, out_idx| unsafe {
                    let a = *(a_ptr.offset(a_off) as *const u64);
                    let b = *(b_ptr.offset(b_off) as *const u64);
                    let out = out_ptr as *mut u64;
                    *out.add(out_idx) = a $rust_op b;
                },
            );
        };
    }

    register_u64_binary!(BinaryOp::Add, +);
    register_u64_binary!(BinaryOp::Sub, -);
    register_u64_binary!(BinaryOp::Mul, *);
    register_u64_binary!(BinaryOp::Div, /);

    // Register uint32 binary loops
    macro_rules! register_u32_binary {
        ($op:expr, $rust_op:tt) => {
            reg.register_binary(
                $op,
                TypeSignature::binary(DTypeKind::Uint32, DTypeKind::Uint32, DTypeKind::Uint32),
                |a_ptr, a_off, b_ptr, b_off, out_ptr, out_idx| unsafe {
                    let a = *(a_ptr.offset(a_off) as *const u32);
                    let b = *(b_ptr.offset(b_off) as *const u32);
                    let out = out_ptr as *mut u32;
                    *out.add(out_idx) = a $rust_op b;
                },
            );
        };
    }

    register_u32_binary!(BinaryOp::Add, +);
    register_u32_binary!(BinaryOp::Sub, -);
    register_u32_binary!(BinaryOp::Mul, *);
    register_u32_binary!(BinaryOp::Div, /);

    // Register uint8 binary loops
    macro_rules! register_u8_binary {
        ($op:expr, $rust_op:tt) => {
            reg.register_binary(
                $op,
                TypeSignature::binary(DTypeKind::Uint8, DTypeKind::Uint8, DTypeKind::Uint8),
                |a_ptr, a_off, b_ptr, b_off, out_ptr, out_idx| unsafe {
                    let a = *(a_ptr.offset(a_off) as *const u8);
                    let b = *(b_ptr.offset(b_off) as *const u8);
                    let out = out_ptr as *mut u8;
                    *out.add(out_idx) = a $rust_op b;
                },
            );
        };
    }

    register_u8_binary!(BinaryOp::Add, +);
    register_u8_binary!(BinaryOp::Sub, -);
    register_u8_binary!(BinaryOp::Mul, *);
    register_u8_binary!(BinaryOp::Div, /);

    // Register bool binary loops (using logical ops for add/mul)
    reg.register_binary(
        BinaryOp::Add,
        TypeSignature::binary(DTypeKind::Bool, DTypeKind::Bool, DTypeKind::Bool),
        |a_ptr, a_off, b_ptr, b_off, out_ptr, out_idx| unsafe {
            let a = *(a_ptr.offset(a_off) as *const u8) != 0;
            let b = *(b_ptr.offset(b_off) as *const u8) != 0;
            *(out_ptr as *mut u8).add(out_idx) = (a || b) as u8;
        },
    );
    reg.register_binary(
        BinaryOp::Mul,
        TypeSignature::binary(DTypeKind::Bool, DTypeKind::Bool, DTypeKind::Bool),
        |a_ptr, a_off, b_ptr, b_off, out_ptr, out_idx| unsafe {
            let a = *(a_ptr.offset(a_off) as *const u8) != 0;
            let b = *(b_ptr.offset(b_off) as *const u8) != 0;
            *(out_ptr as *mut u8).add(out_idx) = (a && b) as u8;
        },
    );

    // Register complex128 binary loops
    reg.register_binary(
        BinaryOp::Add,
        TypeSignature::binary(DTypeKind::Complex128, DTypeKind::Complex128, DTypeKind::Complex128),
        |a_ptr, a_off, b_ptr, b_off, out_ptr, out_idx| unsafe {
            let a = a_ptr.offset(a_off) as *const f64;
            let b = b_ptr.offset(b_off) as *const f64;
            let out = (out_ptr as *mut f64).add(out_idx * 2);
            *out = *a + *b;
            *out.add(1) = *a.add(1) + *b.add(1);
        },
    );
    reg.register_binary(
        BinaryOp::Sub,
        TypeSignature::binary(DTypeKind::Complex128, DTypeKind::Complex128, DTypeKind::Complex128),
        |a_ptr, a_off, b_ptr, b_off, out_ptr, out_idx| unsafe {
            let a = a_ptr.offset(a_off) as *const f64;
            let b = b_ptr.offset(b_off) as *const f64;
            let out = (out_ptr as *mut f64).add(out_idx * 2);
            *out = *a - *b;
            *out.add(1) = *a.add(1) - *b.add(1);
        },
    );
    reg.register_binary(
        BinaryOp::Mul,
        TypeSignature::binary(DTypeKind::Complex128, DTypeKind::Complex128, DTypeKind::Complex128),
        |a_ptr, a_off, b_ptr, b_off, out_ptr, out_idx| unsafe {
            let a = a_ptr.offset(a_off) as *const f64;
            let b = b_ptr.offset(b_off) as *const f64;
            let ar = *a; let ai = *a.add(1);
            let br = *b; let bi = *b.add(1);
            let out = (out_ptr as *mut f64).add(out_idx * 2);
            *out = ar * br - ai * bi;
            *out.add(1) = ar * bi + ai * br;
        },
    );
    reg.register_binary(
        BinaryOp::Div,
        TypeSignature::binary(DTypeKind::Complex128, DTypeKind::Complex128, DTypeKind::Complex128),
        |a_ptr, a_off, b_ptr, b_off, out_ptr, out_idx| unsafe {
            let a = a_ptr.offset(a_off) as *const f64;
            let b = b_ptr.offset(b_off) as *const f64;
            let ar = *a; let ai = *a.add(1);
            let br = *b; let bi = *b.add(1);
            let denom = br * br + bi * bi;
            let out = (out_ptr as *mut f64).add(out_idx * 2);
            if denom != 0.0 {
                *out = (ar * br + ai * bi) / denom;
                *out.add(1) = (ai * br - ar * bi) / denom;
            } else {
                *out = f64::NAN;
                *out.add(1) = f64::NAN;
            }
        },
    );

    // ========================================================================
    // Unary loops
    // ========================================================================

    // f64 unary loops
    reg.register_unary(
        UnaryOp::Neg,
        TypeSignature::unary(DTypeKind::Float64, DTypeKind::Float64),
        |src_ptr, src_off, out_ptr, out_idx| unsafe {
            let v = *(src_ptr.offset(src_off) as *const f64);
            *(out_ptr as *mut f64).add(out_idx) = -v;
        },
    );
    reg.register_unary(
        UnaryOp::Abs,
        TypeSignature::unary(DTypeKind::Float64, DTypeKind::Float64),
        |src_ptr, src_off, out_ptr, out_idx| unsafe {
            let v = *(src_ptr.offset(src_off) as *const f64);
            *(out_ptr as *mut f64).add(out_idx) = v.abs();
        },
    );
    reg.register_unary(
        UnaryOp::Sqrt,
        TypeSignature::unary(DTypeKind::Float64, DTypeKind::Float64),
        |src_ptr, src_off, out_ptr, out_idx| unsafe {
            let v = *(src_ptr.offset(src_off) as *const f64);
            *(out_ptr as *mut f64).add(out_idx) = v.sqrt();
        },
    );
    reg.register_unary(
        UnaryOp::Exp,
        TypeSignature::unary(DTypeKind::Float64, DTypeKind::Float64),
        |src_ptr, src_off, out_ptr, out_idx| unsafe {
            let v = *(src_ptr.offset(src_off) as *const f64);
            *(out_ptr as *mut f64).add(out_idx) = v.exp();
        },
    );
    reg.register_unary(
        UnaryOp::Log,
        TypeSignature::unary(DTypeKind::Float64, DTypeKind::Float64),
        |src_ptr, src_off, out_ptr, out_idx| unsafe {
            let v = *(src_ptr.offset(src_off) as *const f64);
            *(out_ptr as *mut f64).add(out_idx) = v.ln();
        },
    );
    reg.register_unary(
        UnaryOp::Sin,
        TypeSignature::unary(DTypeKind::Float64, DTypeKind::Float64),
        |src_ptr, src_off, out_ptr, out_idx| unsafe {
            let v = *(src_ptr.offset(src_off) as *const f64);
            *(out_ptr as *mut f64).add(out_idx) = v.sin();
        },
    );
    reg.register_unary(
        UnaryOp::Cos,
        TypeSignature::unary(DTypeKind::Float64, DTypeKind::Float64),
        |src_ptr, src_off, out_ptr, out_idx| unsafe {
            let v = *(src_ptr.offset(src_off) as *const f64);
            *(out_ptr as *mut f64).add(out_idx) = v.cos();
        },
    );
    reg.register_unary(
        UnaryOp::Tan,
        TypeSignature::unary(DTypeKind::Float64, DTypeKind::Float64),
        |src_ptr, src_off, out_ptr, out_idx| unsafe {
            let v = *(src_ptr.offset(src_off) as *const f64);
            *(out_ptr as *mut f64).add(out_idx) = v.tan();
        },
    );

    // f32 unary loops
    reg.register_unary(
        UnaryOp::Neg,
        TypeSignature::unary(DTypeKind::Float32, DTypeKind::Float32),
        |src_ptr, src_off, out_ptr, out_idx| unsafe {
            let v = *(src_ptr.offset(src_off) as *const f32);
            *(out_ptr as *mut f32).add(out_idx) = -v;
        },
    );
    reg.register_unary(
        UnaryOp::Abs,
        TypeSignature::unary(DTypeKind::Float32, DTypeKind::Float32),
        |src_ptr, src_off, out_ptr, out_idx| unsafe {
            let v = *(src_ptr.offset(src_off) as *const f32);
            *(out_ptr as *mut f32).add(out_idx) = v.abs();
        },
    );
    reg.register_unary(
        UnaryOp::Sqrt,
        TypeSignature::unary(DTypeKind::Float32, DTypeKind::Float32),
        |src_ptr, src_off, out_ptr, out_idx| unsafe {
            let v = *(src_ptr.offset(src_off) as *const f32);
            *(out_ptr as *mut f32).add(out_idx) = v.sqrt();
        },
    );

    // i64 unary loops
    reg.register_unary(
        UnaryOp::Neg,
        TypeSignature::unary(DTypeKind::Int64, DTypeKind::Int64),
        |src_ptr, src_off, out_ptr, out_idx| unsafe {
            let v = *(src_ptr.offset(src_off) as *const i64);
            *(out_ptr as *mut i64).add(out_idx) = -v;
        },
    );
    reg.register_unary(
        UnaryOp::Abs,
        TypeSignature::unary(DTypeKind::Int64, DTypeKind::Int64),
        |src_ptr, src_off, out_ptr, out_idx| unsafe {
            let v = *(src_ptr.offset(src_off) as *const i64);
            *(out_ptr as *mut i64).add(out_idx) = v.abs();
        },
    );

    // i32 unary loops
    reg.register_unary(
        UnaryOp::Neg,
        TypeSignature::unary(DTypeKind::Int32, DTypeKind::Int32),
        |src_ptr, src_off, out_ptr, out_idx| unsafe {
            let v = *(src_ptr.offset(src_off) as *const i32);
            *(out_ptr as *mut i32).add(out_idx) = -v;
        },
    );
    reg.register_unary(
        UnaryOp::Abs,
        TypeSignature::unary(DTypeKind::Int32, DTypeKind::Int32),
        |src_ptr, src_off, out_ptr, out_idx| unsafe {
            let v = *(src_ptr.offset(src_off) as *const i32);
            *(out_ptr as *mut i32).add(out_idx) = v.abs();
        },
    );

    // uint64 unary loops (Abs only - no Neg for unsigned)
    reg.register_unary(
        UnaryOp::Abs,
        TypeSignature::unary(DTypeKind::Uint64, DTypeKind::Uint64),
        |src_ptr, src_off, out_ptr, out_idx| unsafe {
            let v = *(src_ptr.offset(src_off) as *const u64);
            *(out_ptr as *mut u64).add(out_idx) = v;
        },
    );

    // uint32 unary loops
    reg.register_unary(
        UnaryOp::Abs,
        TypeSignature::unary(DTypeKind::Uint32, DTypeKind::Uint32),
        |src_ptr, src_off, out_ptr, out_idx| unsafe {
            let v = *(src_ptr.offset(src_off) as *const u32);
            *(out_ptr as *mut u32).add(out_idx) = v;
        },
    );

    // uint8 unary loops
    reg.register_unary(
        UnaryOp::Abs,
        TypeSignature::unary(DTypeKind::Uint8, DTypeKind::Uint8),
        |src_ptr, src_off, out_ptr, out_idx| unsafe {
            let v = *(src_ptr.offset(src_off) as *const u8);
            *(out_ptr as *mut u8).add(out_idx) = v;
        },
    );

    // complex128 unary loops
    reg.register_unary(
        UnaryOp::Neg,
        TypeSignature::unary(DTypeKind::Complex128, DTypeKind::Complex128),
        |src_ptr, src_off, out_ptr, out_idx| unsafe {
            let src = src_ptr.offset(src_off) as *const f64;
            let out = (out_ptr as *mut f64).add(out_idx * 2);
            *out = -(*src);
            *out.add(1) = -(*src.add(1));
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

        let a: f64 = 3.0;
        let b: f64 = 4.0;
        let mut out: f64 = 0.0;

        unsafe {
            loop_fn(
                &a as *const f64 as *const u8,
                0,
                &b as *const f64 as *const u8,
                0,
                &mut out as *mut f64 as *mut u8,
                0,
            );
        }

        assert_eq!(out, 7.0);
    }

    #[test]
    fn test_i64_mul_loop() {
        let reg = init_default_loops();
        let (loop_fn, _) = reg
            .lookup_binary(BinaryOp::Mul, DTypeKind::Int64, DTypeKind::Int64)
            .unwrap();

        let a: i64 = 5;
        let b: i64 = 6;
        let mut out: i64 = 0;

        unsafe {
            loop_fn(
                &a as *const i64 as *const u8,
                0,
                &b as *const i64 as *const u8,
                0,
                &mut out as *mut i64 as *mut u8,
                0,
            );
        }

        assert_eq!(out, 30);
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

        let src: f64 = 3.5;
        let mut out: f64 = 0.0;

        unsafe {
            loop_fn(
                &src as *const f64 as *const u8,
                0,
                &mut out as *mut f64 as *mut u8,
                0,
            );
        }

        assert_eq!(out, -3.5);
    }

    #[test]
    fn test_f64_sqrt_loop() {
        let reg = init_default_loops();
        let (loop_fn, _) = reg
            .lookup_unary(UnaryOp::Sqrt, DTypeKind::Float64)
            .unwrap();

        let src: f64 = 16.0;
        let mut out: f64 = 0.0;

        unsafe {
            loop_fn(
                &src as *const f64 as *const u8,
                0,
                &mut out as *mut f64 as *mut u8,
                0,
            );
        }

        assert_eq!(out, 4.0);
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
