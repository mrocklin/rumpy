# Random Number Generation Design

## Overview

The `rp.random` module provides numpy-compatible random number generation
using PCG64DXSM BitGenerator.

## BitGenerator: PCG64DXSM

Custom implementation matching NumPy's PCG64DXSM exactly (DXSM output, cheap multiplier).

Two backends:
- `default_rng(seed)`: Uses rand_pcg for convenience (different seed expansion)
- `Generator.from_numpy_state(state, inc)`: Exact numpy state for bit-identical output

## Algorithms

| Function | Algorithm | Numpy Match |
|----------|-----------|-------------|
| `random()` | 53-bit shift | Exact ✓ |
| `uniform()` | Scale random() | Exact ✓ |
| `integers()` | Lemire's bounded | Statistical only |
| `normal()` | Box-Muller | Statistical only |
| `exponential()` | Inverse transform | Statistical only |

See `designs/deviations.md` for details on algorithm differences.

## API

```python
# Simple usage (reproducible within rumpy)
rng = rp.random.default_rng(seed=42)
rng.random(size=(3, 4))

# Exact numpy matching
bg = np.random.PCG64DXSM(seed)
state, inc = bg.state['state']['state'], bg.state['state']['inc']
rng = rp.random.Generator.from_numpy_state(state, inc)
```
