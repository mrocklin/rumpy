// Random number generation module matching numpy.random
//
// Uses PCG64DXSM BitGenerator (matches numpy.random.PCG64DXSM)

mod generator;
mod pcg64;

pub use generator::Generator;

/// Create a new Generator with PCG64DXSM BitGenerator.
/// Equivalent to numpy.random.default_rng() but uses PCG64DXSM.
pub fn default_rng(seed: u64) -> Generator {
    Generator::new(seed)
}
