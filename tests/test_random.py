"""Tests for rumpy.random module."""
import numpy as np
import rumpy as rp
from helpers import assert_eq


def test_default_rng():
    """Test that default_rng creates a Generator."""
    rng = rp.random.default_rng(42)
    assert rng is not None


def test_random_scalar():
    """Test generating a single random float."""
    rng = rp.random.default_rng(42)
    val = rng.random()
    assert isinstance(val, float)
    assert 0.0 <= val < 1.0


def test_random_array():
    """Test generating array of random floats."""
    rng = rp.random.default_rng(42)
    arr = rng.random(size=(3, 4))
    assert arr.shape == (3, 4)
    # All values should be in [0, 1)
    n = np.asarray(arr)
    assert np.all(n >= 0.0)
    assert np.all(n < 1.0)


def test_random_reproducible():
    """Test that same seed gives same results."""
    rng1 = rp.random.default_rng(12345)
    rng2 = rp.random.default_rng(12345)

    arr1 = rng1.random(size=10)
    arr2 = rng2.random(size=10)

    np.testing.assert_array_equal(np.asarray(arr1), np.asarray(arr2))


def test_integers_scalar():
    """Test generating a single random integer."""
    rng = rp.random.default_rng(42)
    val = rng.integers(10)  # [0, 10)
    assert isinstance(val, int)
    assert 0 <= val < 10


def test_integers_range():
    """Test generating integers in range [low, high)."""
    rng = rp.random.default_rng(42)
    arr = rng.integers(5, 15, size=100)
    n = np.asarray(arr)
    assert np.all(n >= 5)
    assert np.all(n < 15)


def test_uniform():
    """Test uniform distribution in [low, high)."""
    rng = rp.random.default_rng(42)
    arr = rng.uniform(2.0, 5.0, size=100)
    n = np.asarray(arr)
    assert np.all(n >= 2.0)
    assert np.all(n < 5.0)


def test_normal():
    """Test normal distribution has correct mean and std."""
    rng = rp.random.default_rng(42)
    arr = rng.normal(loc=10.0, scale=2.0, size=10000)
    n = np.asarray(arr)
    # Should be approximately correct (statistical test)
    assert abs(n.mean() - 10.0) < 0.1
    assert abs(n.std() - 2.0) < 0.1


def test_standard_normal():
    """Test standard normal has mean 0 and std 1."""
    rng = rp.random.default_rng(42)
    arr = rng.standard_normal(size=10000)
    n = np.asarray(arr)
    assert abs(n.mean()) < 0.05
    assert abs(n.std() - 1.0) < 0.05


def test_exponential():
    """Test exponential distribution."""
    rng = rp.random.default_rng(42)
    arr = rng.exponential(scale=2.0, size=10000)
    n = np.asarray(arr)
    # All values should be positive
    assert np.all(n > 0)
    # Mean should be approximately scale
    assert abs(n.mean() - 2.0) < 0.1


def test_standard_exponential():
    """Test standard exponential has mean 1."""
    rng = rp.random.default_rng(42)
    arr = rng.standard_exponential(size=10000)
    n = np.asarray(arr)
    assert np.all(n > 0)
    assert abs(n.mean() - 1.0) < 0.05


def test_generator_class():
    """Test creating Generator directly."""
    gen = rp.random.Generator(seed=42)
    val = gen.random()
    assert isinstance(val, float)


# Exact value matching tests - using from_numpy_state for identical output

class TestExactMatch:
    """Test exact value matching when using from_numpy_state."""

    def _get_numpy_state(self, seed):
        """Get state/inc from numpy's PCG64DXSM."""
        bg = np.random.PCG64DXSM(seed)
        return bg.state['state']['state'], bg.state['state']['inc']

    def test_random_exact(self):
        """Test random() produces exactly same values as numpy."""
        seed = 42
        state, inc = self._get_numpy_state(seed)

        rng_rp = rp.random.Generator.from_numpy_state(state, inc)
        rng_np = np.random.Generator(np.random.PCG64DXSM(seed))

        r = rng_rp.random(size=100)
        n = rng_np.random(size=100)
        assert_eq(r, n)

    def test_random_2d_exact(self):
        """Test random() with 2D shape produces exact match."""
        seed = 123
        state, inc = self._get_numpy_state(seed)

        rng_rp = rp.random.Generator.from_numpy_state(state, inc)
        rng_np = np.random.Generator(np.random.PCG64DXSM(seed))

        r = rng_rp.random(size=(10, 20))
        n = rng_np.random(size=(10, 20))
        assert_eq(r, n)

    def test_uniform_exact(self):
        """Test uniform() produces exactly same values as numpy."""
        seed = 456
        state, inc = self._get_numpy_state(seed)

        rng_rp = rp.random.Generator.from_numpy_state(state, inc)
        rng_np = np.random.Generator(np.random.PCG64DXSM(seed))

        r = rng_rp.uniform(5.0, 10.0, size=100)
        n = rng_np.uniform(5.0, 10.0, size=100)
        assert_eq(r, n)

    def test_integers_range(self):
        """Test integers() produces values in correct range (algorithm may differ)."""
        # Note: Numpy uses a different bounded integer algorithm than Lemire's,
        # so we test range/distribution rather than exact values.
        seed = 789
        state, inc = self._get_numpy_state(seed)

        rng_rp = rp.random.Generator.from_numpy_state(state, inc)

        r = rng_rp.integers(0, 100, size=1000)
        r_np = np.asarray(r)

        # Should be in range [0, 100)
        assert r_np.min() >= 0
        assert r_np.max() < 100
        # Should be roughly uniform
        assert 40 < r_np.mean() < 60


# Distribution comparison tests - compare statistical properties
# Since seeding differs, we verify distributions match numpy's behavior

class TestRandomVsNumpy:
    """Compare rumpy.random distributions against numpy."""

    def test_random_shape(self):
        """Test random() produces same shape as numpy."""
        rng_rp = rp.random.default_rng(42)
        rng_np = np.random.default_rng(42)

        r = rng_rp.random(size=(3, 4, 5))
        n = rng_np.random(size=(3, 4, 5))

        assert np.asarray(r).shape == n.shape

    def test_integers_shape_and_range(self):
        """Test integers() matches numpy shape and range."""
        rng_rp = rp.random.default_rng(42)
        rng_np = np.random.default_rng(42)

        r = rng_rp.integers(5, 15, size=(100,))
        n = rng_np.integers(5, 15, size=(100,))

        r_np = np.asarray(r)
        assert r_np.shape == n.shape
        # Same range constraints
        assert r_np.min() >= 5
        assert r_np.max() < 15
        assert n.min() >= 5
        assert n.max() < 15

    def test_uniform_shape_and_range(self):
        """Test uniform() matches numpy shape and range."""
        rng_rp = rp.random.default_rng(42)
        rng_np = np.random.default_rng(42)

        r = rng_rp.uniform(2.0, 8.0, size=(50, 50))
        n = rng_np.uniform(2.0, 8.0, size=(50, 50))

        r_np = np.asarray(r)
        assert r_np.shape == n.shape
        # Same range constraints
        assert r_np.min() >= 2.0
        assert r_np.max() < 8.0

    def test_normal_statistics(self):
        """Test normal() produces same statistical properties as numpy."""
        loc, scale = 5.0, 2.0
        size = (10000,)

        rng_rp = rp.random.default_rng(42)
        rng_np = np.random.default_rng(42)

        r = rng_rp.normal(loc, scale, size=size)
        n = rng_np.normal(loc, scale, size=size)

        r_np = np.asarray(r)

        # Both should have approximately the same mean and std
        assert abs(r_np.mean() - loc) < 0.1
        assert abs(n.mean() - loc) < 0.1
        assert abs(r_np.std() - scale) < 0.1
        assert abs(n.std() - scale) < 0.1

    def test_standard_normal_statistics(self):
        """Test standard_normal() matches numpy statistics."""
        size = (10000,)

        rng_rp = rp.random.default_rng(42)
        rng_np = np.random.default_rng(42)

        r = rng_rp.standard_normal(size=size)
        n = rng_np.standard_normal(size=size)

        r_np = np.asarray(r)

        # Both should have mean ~0, std ~1
        assert abs(r_np.mean()) < 0.05
        assert abs(n.mean()) < 0.05
        assert abs(r_np.std() - 1.0) < 0.05
        assert abs(n.std() - 1.0) < 0.05

    def test_exponential_statistics(self):
        """Test exponential() matches numpy statistics."""
        scale = 3.0
        size = (10000,)

        rng_rp = rp.random.default_rng(42)
        rng_np = np.random.default_rng(42)

        r = rng_rp.exponential(scale, size=size)
        n = rng_np.exponential(scale, size=size)

        r_np = np.asarray(r)

        # Both should be positive with mean ~scale
        assert r_np.min() > 0
        assert n.min() > 0
        assert abs(r_np.mean() - scale) < 0.2
        assert abs(n.mean() - scale) < 0.2

    def test_standard_exponential_statistics(self):
        """Test standard_exponential() matches numpy statistics."""
        size = (10000,)

        rng_rp = rp.random.default_rng(42)
        rng_np = np.random.default_rng(42)

        r = rng_rp.standard_exponential(size=size)
        n = rng_np.standard_exponential(size=size)

        r_np = np.asarray(r)

        # Both should be positive with mean ~1
        assert r_np.min() > 0
        assert n.min() > 0
        assert abs(r_np.mean() - 1.0) < 0.05
        assert abs(n.mean() - 1.0) < 0.05
