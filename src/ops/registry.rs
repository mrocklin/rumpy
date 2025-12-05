//! UFunc registry for type-specific inner loops.
//!
//! Provides NumPy-style dispatch: registered loops checked first,
//! then fallback to DTypeOps trait methods.

use crate::array::dtype::{BinaryOp, DTypeKind, UnaryOp};
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

/// Registry for ufunc inner loops.
pub struct UFuncRegistry {
    binary_loops: HashMap<(BinaryOp, TypeSignature), BinaryLoopFn>,
    unary_loops: HashMap<(UnaryOp, TypeSignature), UnaryLoopFn>,
}

impl UFuncRegistry {
    pub fn new() -> Self {
        Self {
            binary_loops: HashMap::new(),
            unary_loops: HashMap::new(),
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
}
