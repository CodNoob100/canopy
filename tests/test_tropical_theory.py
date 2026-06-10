"""
Tests for canopy.math.tropical_theory
======================================

Verifies:
1. TropicalPolynomial operations (tropical sum, conventional sum, scale, prune)
2. TropicalRational operations (evaluate, relu)
3. Layer propagation against hand-computed results
4. Full architecture-to-tropical map for small known networks
5. Architecture-level generic_linear_region_count against published formulas
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from canopy.math.tropical_theory import (
    TropicalPolynomial,
    TropicalRational,
    propagate_linear,
    propagate_relu_layer,
    architecture_to_tropical,
    generic_linear_region_count,
    linear_region_upper_bound_tight,
)


# ===================================================================
#  TropicalPolynomial — construction and properties
# ===================================================================

class TestTropicalPolynomialBasics:
    def test_zero(self):
        p = TropicalPolynomial.zero(3)
        assert p.num_terms == 1
        assert p.input_dim == 3
        assert p.evaluate(np.array([1.0, 2.0, 3.0])) == 0.0

    def test_constant(self):
        p = TropicalPolynomial.constant(5.0, 2)
        assert p.evaluate(np.array([100.0, -200.0])) == 5.0

    def test_identity(self):
        p = TropicalPolynomial.identity(1, 3)
        x = np.array([10.0, 42.0, -5.0])
        assert p.evaluate(x) == 42.0

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError):
            TropicalPolynomial(
                coefficients=np.array([1.0, 2.0]),
                exponents=np.array([[1.0]]),  # wrong shape
            )


# ===================================================================
#  TropicalPolynomial — evaluation
# ===================================================================

class TestTropicalPolynomialEval:
    def test_max_of_affine(self):
        """p(x) = max(3 + 2x, 1 + 5x)"""
        p = TropicalPolynomial(
            coefficients=np.array([3.0, 1.0]),
            exponents=np.array([[2.0], [5.0]]),
        )
        # At x=0: max(3, 1) = 3
        assert p.evaluate(np.array([0.0])) == 3.0
        # At x=1: max(5, 6) = 6
        assert p.evaluate(np.array([1.0])) == 6.0
        # At x=-1: max(1, -4) = 1
        assert p.evaluate(np.array([-1.0])) == 1.0

    def test_batch_eval(self):
        p = TropicalPolynomial(
            coefficients=np.array([0.0, 1.0]),
            exponents=np.array([[1.0], [0.5]]),
        )
        X = np.array([[0.0], [2.0], [4.0]])
        result = p.evaluate_batch(X)
        expected = np.array([1.0, 2.0, 4.0])
        assert_allclose(result, expected)

    def test_active_term(self):
        p = TropicalPolynomial(
            coefficients=np.array([3.0, 1.0]),
            exponents=np.array([[2.0], [5.0]]),
        )
        # At x=0: term 0 wins (3 > 1)
        assert p.active_term(np.array([0.0])) == 0
        # At x=10: term 1 wins (23 vs 51)
        assert p.active_term(np.array([10.0])) == 1


# ===================================================================
#  TropicalPolynomial — operations
# ===================================================================

class TestTropicalPolynomialOps:
    def test_tropical_sum_union(self):
        """⊕ is union of terms."""
        p = TropicalPolynomial(
            coefficients=np.array([1.0]),
            exponents=np.array([[1.0, 0.0]]),
        )
        q = TropicalPolynomial(
            coefficients=np.array([2.0]),
            exponents=np.array([[0.0, 1.0]]),
        )
        r = p.tropical_sum(q)
        assert r.num_terms == 2

    def test_tropical_sum_semantics(self):
        """(p ⊕ q)(x) = max(p(x), q(x))."""
        p = TropicalPolynomial(
            coefficients=np.array([0.0]),
            exponents=np.array([[1.0]]),
        )
        q = TropicalPolynomial(
            coefficients=np.array([2.0]),
            exponents=np.array([[0.0]]),
        )
        r = p.tropical_sum(q)
        # At x=0: max(0, 2) = 2
        assert r.evaluate(np.array([0.0])) == 2.0
        # At x=3: max(3, 2) = 3
        assert r.evaluate(np.array([3.0])) == 3.0

    def test_conventional_sum_cartesian(self):
        """p + q: Cartesian product of terms."""
        p = TropicalPolynomial(
            coefficients=np.array([1.0, 2.0]),
            exponents=np.array([[1.0], [0.0]]),
        )
        q = TropicalPolynomial(
            coefficients=np.array([3.0, 4.0]),
            exponents=np.array([[0.0], [1.0]]),
        )
        r = p.conventional_sum(q)
        assert r.num_terms == 4  # 2 × 2

    def test_conventional_sum_semantics(self):
        """(p + q)(x) = p(x) + q(x)  in conventional arithmetic."""
        p = TropicalPolynomial(
            coefficients=np.array([0.0, 1.0]),
            exponents=np.array([[1.0], [0.0]]),
        )
        q = TropicalPolynomial(
            coefficients=np.array([0.0, 2.0]),
            exponents=np.array([[1.0], [0.0]]),
        )
        r = p.conventional_sum(q)
        for x_val in [-2.0, 0.0, 0.5, 1.0, 3.0]:
            x = np.array([x_val])
            expected = p.evaluate(x) + q.evaluate(x)
            assert_allclose(r.evaluate(x), expected, atol=1e-12)

    def test_scale_positive(self):
        p = TropicalPolynomial(
            coefficients=np.array([2.0, 3.0]),
            exponents=np.array([[1.0], [0.0]]),
        )
        r = p.scale(3.0)
        assert r.num_terms == 2
        assert_allclose(r.coefficients, [6.0, 9.0])
        assert_allclose(r.exponents, [[3.0], [0.0]])

    def test_scale_zero(self):
        p = TropicalPolynomial(
            coefficients=np.array([2.0]),
            exponents=np.array([[1.0]]),
        )
        r = p.scale(0.0)
        assert r.num_terms == 1
        assert r.evaluate(np.array([100.0])) == 0.0

    def test_scale_negative_raises(self):
        p = TropicalPolynomial.zero(1)
        with pytest.raises(ValueError):
            p.scale(-1.0)

    def test_shift(self):
        p = TropicalPolynomial(
            coefficients=np.array([1.0, 2.0]),
            exponents=np.array([[1.0], [0.0]]),
        )
        r = p.shift(5.0)
        assert_allclose(r.coefficients, [6.0, 7.0])
        # Exponents unchanged
        assert_allclose(r.exponents, p.exponents)

    def test_dim_mismatch_tropical_sum_raises(self):
        p = TropicalPolynomial.zero(2)
        q = TropicalPolynomial.zero(3)
        with pytest.raises(ValueError):
            p.tropical_sum(q)


# ===================================================================
#  TropicalPolynomial — deduplication and pruning
# ===================================================================

class TestTropicalPolynomialPrune:
    def test_deduplicate(self):
        """Merge duplicate exponents, keep highest coefficient."""
        p = TropicalPolynomial(
            coefficients=np.array([1.0, 3.0, 2.0]),
            exponents=np.array([[1.0], [1.0], [0.0]]),
        )
        r = p.deduplicate()
        assert r.num_terms == 2
        # The two [1.0] exponents should merge, keeping coeff=3.0
        idx_1 = np.where(np.abs(r.exponents[:, 0] - 1.0) < 1e-12)[0][0]
        assert r.coefficients[idx_1] == 3.0

    def test_prune_dominated(self):
        """A term that is always below others should be removed."""
        # p(x) = max(x, -x, -100)
        # The term -100 (constant) is dominated whenever |x| ≥ 100,
        # but actually it's dominated for all x since max(x, -x) ≥ 0 > -100.
        p = TropicalPolynomial(
            coefficients=np.array([0.0, 0.0, -100.0]),
            exponents=np.array([[1.0], [-1.0], [0.0]]),
        )
        r = p.prune_dominated()
        assert r.num_terms == 2

    def test_prune_keeps_all_active(self):
        """When all terms are active, none are pruned."""
        # p(x) = max(0, x)  — both terms are needed
        p = TropicalPolynomial(
            coefficients=np.array([0.0, 0.0]),
            exponents=np.array([[0.0], [1.0]]),
        )
        r = p.prune_dominated()
        assert r.num_terms == 2


# ===================================================================
#  TropicalRational — basics
# ===================================================================

class TestTropicalRational:
    def test_from_variable(self):
        f = TropicalRational.from_variable(0, 2)
        x = np.array([3.0, 7.0])
        assert f.evaluate(x) == 3.0

    def test_from_constant_positive(self):
        f = TropicalRational.from_constant(5.0, 1)
        assert f.evaluate(np.array([0.0])) == 5.0
        assert f.evaluate(np.array([100.0])) == 5.0

    def test_from_constant_negative(self):
        f = TropicalRational.from_constant(-3.0, 1)
        assert f.evaluate(np.array([0.0])) == -3.0

    def test_relu_positive_region(self):
        """ReLU of a positive value stays positive."""
        # f(x) = x - 0 = x.  relu(f) = max(0, x).
        f = TropicalRational.from_variable(0, 1)
        g = f.relu()
        assert g.evaluate(np.array([5.0])) == 5.0

    def test_relu_negative_region(self):
        """ReLU of a negative value becomes 0."""
        f = TropicalRational.from_variable(0, 1)
        g = f.relu()
        assert g.evaluate(np.array([-5.0])) == 0.0

    def test_relu_identity(self):
        """
        Verify the theorem: max(0, p - q) = (p ⊕ q) - q.

        Test over many random points.
        """
        rng = np.random.default_rng(42)
        p = TropicalPolynomial(
            coefficients=rng.standard_normal(3),
            exponents=rng.standard_normal((3, 2)),
        )
        q = TropicalPolynomial(
            coefficients=rng.standard_normal(2),
            exponents=rng.standard_normal((2, 2)),
        )
        f = TropicalRational(numerator=p, denominator=q)
        g = f.relu()

        for _ in range(100):
            x = rng.standard_normal(2)
            expected = max(0.0, f.evaluate(x))
            actual = g.evaluate(x)
            assert_allclose(actual, expected, atol=1e-10)

    def test_max_linear_regions_bound(self):
        p = TropicalPolynomial(
            coefficients=np.zeros(4),
            exponents=np.eye(4, 2),
        )
        q = TropicalPolynomial(
            coefficients=np.zeros(3),
            exponents=np.eye(3, 2),
        )
        f = TropicalRational(numerator=p, denominator=q)
        assert f.max_linear_regions == 12  # 4 × 3


# ===================================================================
#  Layer propagation — single hidden layer
# ===================================================================

class TestLayerPropagation:
    def test_single_neuron_relu(self):
        """
        Network: f(x) = ReLU(2x + 1)

        Expected tropical rational:
            numerator = max(0, 1 + 2x)  — 2 terms
            denominator = max(0) = 0     — 1 term
        """
        inputs = [TropicalRational.from_variable(0, 1)]
        W = np.array([[2.0]])
        b = np.array([1.0])

        outputs = propagate_relu_layer(inputs, W, b)
        assert len(outputs) == 1
        f = outputs[0]

        # Check term counts
        assert f.numerator.num_terms == 2
        assert f.denominator.num_terms == 1

        # Check evaluation
        assert_allclose(f.evaluate(np.array([3.0])), 7.0)  # ReLU(7) = 7
        assert_allclose(f.evaluate(np.array([-1.0])), 0.0)  # ReLU(-1) = 0

    def test_two_neurons_one_hidden_layer(self):
        """
        Network:
            h1 = ReLU(x + 1)
            h2 = ReLU(-x + 2)
            out = h1 + h2 (both weight = 1, bias = 0)

        Expected: piecewise linear with up to 3 regions.
        """
        input_dim = 1
        # Hidden layer
        W1 = np.array([[1.0], [-1.0]])  # 2 neurons, 1 input
        b1 = np.array([1.0, 2.0])

        # Output layer (linear, no ReLU)
        W2 = np.array([[1.0, 1.0]])  # 1 output, 2 inputs
        b2 = np.array([0.0])

        outputs = architecture_to_tropical(
            [W1, W2], [b1, b2], input_dim,
        )
        f = outputs[0]

        # Verify at specific points
        # x = 0: h1 = ReLU(1) = 1, h2 = ReLU(2) = 2, out = 3
        assert_allclose(f.evaluate(np.array([0.0])), 3.0, atol=1e-10)
        # x = 5: h1 = ReLU(6) = 6, h2 = ReLU(-3) = 0, out = 6
        assert_allclose(f.evaluate(np.array([5.0])), 6.0, atol=1e-10)
        # x = -5: h1 = ReLU(-4) = 0, h2 = ReLU(7) = 7, out = 7
        assert_allclose(f.evaluate(np.array([-5.0])), 7.0, atol=1e-10)

    def test_negative_weights_in_output(self):
        """
        Network:
            h = ReLU(x + 1)
            out = -2 * h + 3

        f(x) = -2 * max(0, x+1) + 3
        """
        inputs = [TropicalRational.from_variable(0, 1)]

        # Hidden layer with ReLU
        W1 = np.array([[1.0]])
        b1 = np.array([1.0])
        hidden = propagate_relu_layer(inputs, W1, b1)

        # Output layer (linear)
        W2 = np.array([[-2.0]])
        b2 = np.array([3.0])
        outputs = propagate_linear(hidden, W2, b2)
        f = outputs[0]

        # x = 0: h = 1, out = -2 + 3 = 1
        assert_allclose(f.evaluate(np.array([0.0])), 1.0, atol=1e-10)
        # x = -2: h = 0, out = 0 + 3 = 3
        assert_allclose(f.evaluate(np.array([-2.0])), 3.0, atol=1e-10)
        # x = 4: h = 5, out = -10 + 3 = -7
        assert_allclose(f.evaluate(np.array([4.0])), -7.0, atol=1e-10)


# ===================================================================
#  Full architecture_to_tropical — end-to-end
# ===================================================================

class TestArchitectureToTropical:
    def test_single_layer_linear(self):
        """A network with no hidden layers is purely affine: 1 region."""
        W = np.array([[2.0, -1.0]])
        b = np.array([3.0])
        outputs = architecture_to_tropical([W], [b], input_dim=2)
        f = outputs[0]

        x = np.array([1.0, 2.0])
        expected = 2 * 1.0 - 1.0 * 2.0 + 3.0  # = 3
        assert_allclose(f.evaluate(x), expected, atol=1e-10)

        # Only 1 term in num and den → 1 linear region
        assert f.max_linear_regions == 1

    def test_two_layer_matches_manual(self):
        """
        Manual verification for a 1→2→1 network.

        W1 = [[1], [-1]],  b1 = [0, 0]
        W2 = [[1, 1]],     b2 = [0]

        f(x) = ReLU(x) + ReLU(-x) = |x|
        """
        W1 = np.array([[1.0], [-1.0]])
        b1 = np.array([0.0, 0.0])
        W2 = np.array([[1.0, 1.0]])
        b2 = np.array([0.0])

        outputs = architecture_to_tropical([W1, W2], [b1, b2], input_dim=1)
        f = outputs[0]

        # |x| at various points
        for x_val in [-3.0, -1.0, 0.0, 1.0, 3.0]:
            assert_allclose(
                f.evaluate(np.array([x_val])),
                abs(x_val),
                atol=1e-10,
                err_msg=f"Failed at x={x_val}",
            )

    def test_prune_per_layer(self):
        """Verify that pruning per layer produces the same function values."""
        rng = np.random.default_rng(123)
        W1 = rng.standard_normal((3, 2))
        b1 = rng.standard_normal(3)
        W2 = rng.standard_normal((1, 3))
        b2 = rng.standard_normal(1)

        out_no_prune = architecture_to_tropical(
            [W1, W2], [b1, b2], input_dim=2, prune_per_layer=False,
        )
        out_prune = architecture_to_tropical(
            [W1, W2], [b1, b2], input_dim=2, prune_per_layer=True,
        )

        for _ in range(50):
            x = rng.standard_normal(2)
            assert_allclose(
                out_no_prune[0].evaluate(x),
                out_prune[0].evaluate(x),
                atol=1e-8,
            )

    def test_with_relu_on_last(self):
        """apply_relu_to_last=True wraps the output in ReLU."""
        W = np.array([[1.0]])
        b = np.array([-2.0])
        outputs = architecture_to_tropical(
            [W], [b], input_dim=1, apply_relu_to_last=True,
        )
        f = outputs[0]

        # ReLU(x - 2)
        assert_allclose(f.evaluate(np.array([5.0])), 3.0, atol=1e-10)
        assert_allclose(f.evaluate(np.array([0.0])), 0.0, atol=1e-10)


# ===================================================================
#  Generic linear region count — Zaslavsky/Montúfar
# ===================================================================

class TestGenericRegionCount:
    def test_linear_network(self):
        """No hidden layers → 1 region."""
        assert generic_linear_region_count([5, 1]) == 1

    def test_single_hidden_1d_input(self):
        """
        1D input, hidden layer of width 4:
        Σ_{j=0}^{min(4,1)} C(4,j) = C(4,0) + C(4,1) = 1 + 4 = 5
        """
        assert generic_linear_region_count([1, 4, 1]) == 5

    def test_single_hidden_2d_input(self):
        """
        2D input, hidden layer of width 3:
        Σ_{j=0}^{min(3,2)} C(3,j) = C(3,0) + C(3,1) + C(3,2) = 1+3+3 = 7
        """
        assert generic_linear_region_count([2, 3, 1]) == 7

    def test_two_hidden_layers(self):
        """
        1D input, two hidden layers of width 2:
        Layer 1: Σ_{j=0}^{1} C(2,j) = 1+2 = 3
        Layer 2: Σ_{j=0}^{1} C(2,j) = 1+2 = 3
        Product = 9

        Wait — n_0 = 1, so min(n_l, n_0) = 1 for both layers.
        Layer 1: C(2,0) + C(2,1) = 1 + 2 = 3
        Layer 2: C(2,0) + C(2,1) = 1 + 2 = 3
        Product = 9

        But the docstring says generic_linear_region_count([1,2,2,1]) = 8.
        Let me recalculate...

        Actually: for 1D input, 2 neurons → Σ_{j=0}^{1} C(2,j) = 3.
        Second hidden layer of 2: same = 3.
        Product = 9. The docstring example of 8 is wrong. Let me verify.

        Hmm, actually from Montúfar 2014, for 1→2→2→1:
        Upper bound = Π_{l} Σ C(n_l, j) where j goes to min(n_l, n_0) = 1.
        = (C(2,0)+C(2,1)) × (C(2,0)+C(2,1)) = 3 × 3 = 9.

        So the expected answer is 9.
        """
        assert generic_linear_region_count([1, 2, 2, 1]) == 9

    def test_bottleneck(self):
        """
        2D input → hidden of 5 → output:
        Σ_{j=0}^{2} C(5,j) = 1 + 5 + 10 = 16
        """
        assert generic_linear_region_count([2, 5, 1]) == 16

    def test_wide_single_layer_1d(self):
        """
        1D input, 100 hidden neurons:
        Σ_{j=0}^{1} C(100,j) = 1 + 100 = 101
        """
        assert generic_linear_region_count([1, 100, 1]) == 101

    def test_tight_bound_with_bottleneck(self):
        """
        The tight bound accounts for bottleneck layers reducing
        effective dimension.

        3D input → 2 neurons (bottleneck) → 10 neurons → output
        Standard: (C(2,0)+C(2,1)+C(2,2)) × (C(10,0)+C(10,1)+C(10,2)+C(10,3))
                = 4 × (1+10+45+120) = 4 × 176 = 704
        Tight:    (C(2,0)+C(2,1)+C(2,2)) × (C(10,0)+C(10,1)+C(10,2))
                = 4 × (1+10+45) = 4 × 56 = 224
        Because the bottleneck reduces effective dim from 3 to 2.
        """
        standard = generic_linear_region_count([3, 2, 10, 1])
        tight = linear_region_upper_bound_tight([3, 2, 10, 1])
        assert standard == 704
        assert tight == 224
        assert tight <= standard
