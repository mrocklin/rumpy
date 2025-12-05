// PCG64DXSM implementation matching numpy exactly
//
// NumPy's PCG64DXSM uses:
// - 128-bit state
// - 128-bit increment (odd)
// - DXSM output function (double xorshift multiply)
// - Cheap multiplier variant (CM)

/// PCG64DXSM matching numpy's implementation exactly.
pub struct Pcg64Dxsm {
    state: u128,
    increment: u128,
}

// PCG cheap multiplier (same as numpy)
const PCG_CHEAP_MULTIPLIER_128: u128 = 0xda942042e4dd58b5;

impl Pcg64Dxsm {
    /// Create from numpy's state dict values.
    /// state and inc come directly from: bg.state['state']['state'], bg.state['state']['inc']
    pub fn from_numpy_state(state: u128, inc: u128) -> Self {
        Self {
            state,
            increment: inc,
        }
    }

    /// Advance the LCG state.
    #[inline]
    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(PCG_CHEAP_MULTIPLIER_128)
            .wrapping_add(self.increment);
    }

    /// DXSM output function (matching numpy exactly).
    /// Algorithm from numpy's pcg64.h:
    /// lo |= 1; hi ^= hi >> 32; hi *= MULT; hi ^= hi >> 48; hi *= lo;
    #[inline]
    fn output(state: u128) -> u64 {
        let mut hi = (state >> 64) as u64;
        let lo = (state as u64) | 1;  // Force lowest bit to 1

        hi ^= hi >> 32;
        hi = hi.wrapping_mul(0xda942042e4dd58b5);
        hi ^= hi >> 48;
        hi = hi.wrapping_mul(lo);
        hi
    }

    /// Generate next u64.
    pub fn next_u64(&mut self) -> u64 {
        // Output then step (same as numpy)
        let out = Self::output(self.state);
        self.step();
        out
    }
}

impl rand_core::RngCore for Pcg64Dxsm {
    fn next_u32(&mut self) -> u32 {
        Pcg64Dxsm::next_u64(self) as u32
    }

    fn next_u64(&mut self) -> u64 {
        Pcg64Dxsm::next_u64(self)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        rand_core::impls::fill_bytes_via_next(self, dest)
    }
}
