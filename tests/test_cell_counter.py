"""
Tests for canopy.geometry.cell_counter
=======================================

Verifies:
1. Activation pattern utilities
2. ActivationPatternCounter exact results against known networks
3. LayerWiseCounter matches exact counter
4. SamplingCounter produces reasonable lower bounds
5. Unified count_linear_regions auto-dispatch
6. Cross-validation: all methods agree on small networks
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from canopy.geometry.cell_counter import (
    ActivationPatternCounter,
    LayerWiseCounter,
    SamplingCounter,
    count_linear_regions,
    get_activation_pattern,
    get_activation_patterns_batch,
    _network_forward,
)


# ===================================================================
#  Activation pattern utilities
# ===================================================================

class TestActivationPatterns:
    def test_single_relu(self):
        """ReLU(x): active for x>0, inactive for x<0."""
        W = [np.array([[1.0]]), np.array([[1.0]])]
        b = [np.array([0.0]), np.array([0.0])]

        pattern_pos = get_activation_pattern(np.array([1.0]), W, b)
        assert pattern_pos == (1,)

        pattern_neg = get_activation_pattern(np.array([-1.0]), W, b)
        assert pattern_neg == (0,)

    def test_two_neurons(self):
        """Two hidden neurons with opposite slopes."""
        W = [np.array([[1.0], [-1.0]]), np.array([[1.0, 1.0]])]
        b = [np.array([0.0, 0.0]), np.array([0.0])]

        # x = 1: neuron 0 active (1>0), neuron 1 inactive (-1<0)
        p = get_activation_pattern(np.array([1.0]), W, b)
        assert p == (1, 0)

        # x = -1: neuron 0 inactive, neuron 1 active
        p = get_activation_pattern(np.array([-1.0]), W, b)
        assert p == (0, 1)

    def test_batch_matches_single(self):
        """Batch computation matches individual calls."""
        rng = np.random.default_rng(42)
        W = [rng.standard_normal((3, 2)), rng.standard_normal((1, 3))]
        b = [rng.standard_normal(3), rng.standard_normal(1)]

        X = rng.standard_normal((20, 2))
        batch = get_activation_patterns_batch(X, W, b)

        for i in range(20):
            single = get_activation_pattern(X[i], W, b)
            assert tuple(batch[i]) == single

    def test_network_forward(self):
        """Forward pass produces correct output."""
        # f(x) = ReLU(2x + 1)   applied once → output
        W = [np.array([[2.0]]), np.array([[1.0]])]
        b = [np.array([1.0]), np.array([0.0])]

        assert_allclose(_network_forward(np.array([3.0]), W, b), 7.0)
        assert_allclose(_network_forward(np.array([-1.0]), W, b), 0.0)


# ===================================================================
#  ActivationPatternCounter — exact
# ===================================================================

class TestExactCounter:
    def test_pure_linear(self):
        """No hidden layers → 1 region."""
        W = [np.array([[2.0, -1.0]])]
        b = [np.array([3.0])]
        counter = ActivationPatternCounter()
        assert counter.count(W, b, input_dim=2) == 1

    def test_single_relu(self):
        """f(x) = ReLU(x): exactly 2 regions."""
        W = [np.array([[1.0]]), np.array([[1.0]])]
        b = [np.array([0.0]), np.array([0.0])]
        counter = ActivationPatternCounter()
        assert counter.count(W, b, input_dim=1) == 2

    def test_abs_value(self):
        """
        f(x) = ReLU(x) + ReLU(-x) = |x|: exactly 2 regions.
        The two neurons create 3 sign vectors but only 2 are feasible
        (cannot have both x>0 and -x>0 simultaneously... wait, actually:
        at x=0 both are on the boundary).

        Sign vectors:
        (1,0): x > 0, -x < 0 → feasible ✓
        (0,1): x < 0, -x > 0 → feasible ✓
        (0,0): x < 0, -x < 0 → impossible ✗
        (1,1): x > 0, -x > 0 → impossible ✗

        So 2 regions. ✓
        """
        W = [np.array([[1.0], [-1.0]]), np.array([[1.0, 1.0]])]
        b = [np.array([0.0, 0.0]), np.array([0.0])]
        counter = ActivationPatternCounter()
        assert counter.count(W, b, input_dim=1) == 2

    def test_two_neurons_general_position(self):
        """
        1D input, 2 hidden neurons with different thresholds.
        h1 = ReLU(x - 1), h2 = ReLU(x - 3)

        Regions: x < 1, 1 < x < 3, x > 3 → 3 regions.

        Sign vectors:
        (0,0): x < 1, x < 3 → feasible (x < 1)
        (1,0): x > 1, x < 3 → feasible (1 < x < 3)
        (1,1): x > 1, x > 3 → feasible (x > 3)
        (0,1): x < 1, x > 3 → impossible

        So 3 regions. ✓
        """
        W = [np.array([[1.0], [1.0]]), np.array([[1.0, 1.0]])]
        b = [np.array([-1.0, -3.0]), np.array([0.0])]
        counter = ActivationPatternCounter()
        assert counter.count(W, b, input_dim=1) == 3

    def test_2d_input_4_neurons(self):
        """
        2D input, 4 hidden neurons in general position.
        Upper bound: Σ_{j=0}^{2} C(4,j) = 1+4+6 = 11.
        With generic weights, should achieve 11.
        """
        rng = np.random.default_rng(42)
        W = [rng.standard_normal((4, 2)), rng.standard_normal((1, 4))]
        b = [rng.standard_normal(4), rng.standard_normal(1)]
        counter = ActivationPatternCounter()
        count = counter.count(W, b, input_dim=2)
        # For generic weights, expect close to 11
        assert 1 <= count <= 16  # max possible for 4 neurons
        # For seed 42, likely achieves the maximum
        assert count >= 8  # reasonable lower bound for generic weights

    def test_refuses_large_network(self):
        """Should raise ValueError for networks exceeding max_neurons."""
        W = [np.eye(25), np.ones((1, 25))]
        b = [np.zeros(25), np.zeros(1)]
        counter = ActivationPatternCounter(max_neurons=20)
        with pytest.raises(ValueError, match="hidden neurons"):
            counter.count(W, b, input_dim=25)


# ===================================================================
#  LayerWiseCounter
# ===================================================================

class TestLayerWiseCounter:
    def test_single_relu(self):
        W = [np.array([[1.0]]), np.array([[1.0]])]
        b = [np.array([0.0]), np.array([0.0])]
        counter = LayerWiseCounter()
        assert counter.count(W, b, input_dim=1) == 2

    def test_abs_value(self):
        W = [np.array([[1.0], [-1.0]]), np.array([[1.0, 1.0]])]
        b = [np.array([0.0, 0.0]), np.array([0.0])]
        counter = LayerWiseCounter()
        assert counter.count(W, b, input_dim=1) == 2

    def test_three_regions(self):
        """Same 3-region network as exact counter test."""
        W = [np.array([[1.0], [1.0]]), np.array([[1.0, 1.0]])]
        b = [np.array([-1.0, -3.0]), np.array([0.0])]
        counter = LayerWiseCounter()
        assert counter.count(W, b, input_dim=1) == 3

    def test_matches_exact_random(self):
        """LayerWise should match exact on small random networks."""
        rng = np.random.default_rng(99)
        for _ in range(5):
            W1 = rng.standard_normal((3, 2))
            b1 = rng.standard_normal(3)
            W2 = rng.standard_normal((1, 3))
            b2 = rng.standard_normal(1)

            exact = ActivationPatternCounter().count([W1, W2], [b1, b2], 2)
            lw = LayerWiseCounter().count([W1, W2], [b1, b2], 2)
            assert exact == lw, f"Mismatch: exact={exact}, layerwise={lw}"

    def test_two_hidden_layers(self):
        """
        1→2→2→1 network.  LayerWise should handle depth.
        """
        rng = np.random.default_rng(7)
        W1 = rng.standard_normal((2, 1))
        b1 = rng.standard_normal(2)
        W2 = rng.standard_normal((2, 2))
        b2 = rng.standard_normal(2)
        W3 = rng.standard_normal((1, 2))
        b3 = rng.standard_normal(1)

        exact = ActivationPatternCounter().count(
            [W1, W2, W3], [b1, b2, b3], 1
        )
        lw = LayerWiseCounter().count(
            [W1, W2, W3], [b1, b2, b3], 1
        )
        assert exact == lw


# ===================================================================
#  SamplingCounter
# ===================================================================

class TestSamplingCounter:
    def test_single_relu(self):
        """Should find at least 2 patterns for ReLU(x)."""
        W = [np.array([[1.0]]), np.array([[1.0]])]
        b = [np.array([0.0]), np.array([0.0])]
        counter = SamplingCounter(n_samples=1000, seed=42)
        result = counter.count_detailed(W, b, input_dim=1)
        assert result.observed == 2
        assert result.chao1_estimate == 2.0

    def test_three_regions(self):
        """Should find all 3 patterns with enough samples."""
        W = [np.array([[1.0], [1.0]]), np.array([[1.0, 1.0]])]
        b = [np.array([-1.0, -3.0]), np.array([0.0])]
        counter = SamplingCounter(n_samples=5000, seed=42)
        result = counter.count_detailed(W, b, input_dim=1)
        assert result.observed == 3

    def test_lower_bound_property(self):
        """Sampling should give a lower bound on exact count."""
        rng = np.random.default_rng(42)
        W1 = rng.standard_normal((4, 2))
        b1 = rng.standard_normal(4)
        W2 = rng.standard_normal((1, 4))
        b2 = rng.standard_normal(1)

        exact = ActivationPatternCounter().count([W1, W2], [b1, b2], 2)
        sampled = SamplingCounter(n_samples=10000, seed=42).count(
            [W1, W2], [b1, b2], 2
        )
        assert sampled >= exact * 0.5  # At least 50% of exact

    def test_coverage_increases_with_samples(self):
        """More samples → higher coverage."""
        W = [np.array([[1.0], [1.0]]), np.array([[1.0, 1.0]])]
        b = [np.array([-1.0, -3.0]), np.array([0.0])]

        r1 = SamplingCounter(n_samples=100, seed=42).count_detailed(W, b, 1)
        r2 = SamplingCounter(n_samples=5000, seed=42).count_detailed(W, b, 1)
        assert r2.coverage >= r1.coverage


# ===================================================================
#  Unified API
# ===================================================================

class TestUnifiedAPI:
    def test_auto_selects_exact(self):
        """Small network → auto should use exact."""
        W = [np.array([[1.0]]), np.array([[1.0]])]
        b = [np.array([0.0]), np.array([0.0])]
        result = count_linear_regions(W, b, input_dim=1)
        assert result == 2

    def test_auto_pure_linear(self):
        W = [np.array([[1.0, 2.0]])]
        b = [np.array([3.0])]
        assert count_linear_regions(W, b, input_dim=2) == 1

    def test_explicit_methods(self):
        """All explicit methods should work for small networks."""
        W = [np.array([[1.0], [1.0]]), np.array([[1.0, 1.0]])]
        b = [np.array([-1.0, -3.0]), np.array([0.0])]

        exact = count_linear_regions(W, b, 1, method="exact")
        lw = count_linear_regions(W, b, 1, method="layerwise")
        samp = count_linear_regions(W, b, 1, method="sampling", n_samples=5000, seed=42)

        assert exact == 3
        assert lw == 3
        assert samp >= 3  # Chao1 may be ≥ observed

    def test_unknown_method_raises(self):
        W = [np.array([[1.0]]), np.array([[1.0]])]
        b = [np.array([0.0]), np.array([0.0])]
        with pytest.raises(ValueError, match="Unknown method"):
            count_linear_regions(W, b, 1, method="magic")


# ===================================================================
#  Cross-validation: exact vs layerwise on multiple random nets
# ===================================================================

class TestCrossValidation:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_exact_vs_layerwise_1hidden(self, seed):
        """1-hidden-layer random network: exact == layerwise."""
        rng = np.random.default_rng(seed)
        n0, n1 = 2, 4
        W = [rng.standard_normal((n1, n0)), rng.standard_normal((1, n1))]
        b = [rng.standard_normal(n1), rng.standard_normal(1)]

        exact = ActivationPatternCounter().count(W, b, n0)
        lw = LayerWiseCounter().count(W, b, n0)
        assert exact == lw, f"seed={seed}: exact={exact} vs layerwise={lw}"

    @pytest.mark.parametrize("seed", [10, 11, 12])
    def test_exact_vs_layerwise_2hidden(self, seed):
        """2-hidden-layer random network: exact == layerwise."""
        rng = np.random.default_rng(seed)
        n0, n1, n2 = 1, 3, 2
        W = [
            rng.standard_normal((n1, n0)),
            rng.standard_normal((n2, n1)),
            rng.standard_normal((1, n2)),
        ]
        b = [rng.standard_normal(n1), rng.standard_normal(n2), rng.standard_normal(1)]

        exact = ActivationPatternCounter().count(W, b, n0)
        lw = LayerWiseCounter().count(W, b, n0)
        assert exact == lw, f"seed={seed}: exact={exact} vs layerwise={lw}"
