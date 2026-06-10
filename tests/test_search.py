"""
Tests for canopy.nas.tropical_score and canopy.nas.search
==========================================================

Verifies:
1. Scoring correctness in all modes
2. Scoring speed meets <100ms target
3. Evolutionary search converges
4. Search finds architectures with monotonically improving fitness
"""

import numpy as np
import pytest
import time

from canopy.nas.architecture import (
    ArchitectureSpec,
    FeedforwardSearchSpace,
    NASBench201SearchSpace,
)
from canopy.nas.tropical_score import (
    tropical_expressivity_score,
    benchmark_scoring_speed,
    score_batch,
)
from canopy.nas.search import EvolutionarySearch


# ===================================================================
#  Tropical Scoring
# ===================================================================

class TestTropicalScore:
    def test_generic_mode(self):
        """Generic mode returns Montúfar bound."""
        arch = ArchitectureSpec(layer_widths=(2, 4, 1))
        score = tropical_expressivity_score(arch, mode="generic")
        # 4 hidden neurons, 2D input: Σ_{j=0}^{2} C(4,j) = 1+4+6 = 11
        assert score == 11

    def test_tight_mode(self):
        """Tight mode returns Serra bound."""
        arch = ArchitectureSpec(layer_widths=(2, 4, 1))
        score = tropical_expressivity_score(arch, mode="tight")
        assert score >= 1
        assert score <= 11  # tight ≤ generic

    def test_generic_deep_network(self):
        """Deeper networks have higher scores."""
        shallow = ArchitectureSpec(layer_widths=(2, 8, 1))
        deep = ArchitectureSpec(layer_widths=(2, 8, 8, 8, 1))
        s_shallow = tropical_expressivity_score(shallow)
        s_deep = tropical_expressivity_score(deep)
        assert s_deep > s_shallow

    def test_wider_is_more_expressive(self):
        """Wider hidden layers → more linear regions."""
        narrow = ArchitectureSpec(layer_widths=(2, 4, 1))
        wide = ArchitectureSpec(layer_widths=(2, 16, 1))
        s_narrow = tropical_expressivity_score(narrow)
        s_wide = tropical_expressivity_score(wide)
        assert s_wide > s_narrow

    def test_weight_aware_mode(self):
        """Weight-aware mode returns a valid count."""
        arch = ArchitectureSpec(layer_widths=(1, 3, 1))
        score = tropical_expressivity_score(
            arch, mode="weight_aware", seed=42
        )
        assert score >= 1

    def test_weight_aware_with_explicit_weights(self):
        """Explicit weights: ReLU(x) → 2 regions."""
        arch = ArchitectureSpec(layer_widths=(1, 1, 1))
        W = [np.array([[1.0]]), np.array([[1.0]])]
        b = [np.array([0.0]), np.array([0.0])]
        score = tropical_expressivity_score(
            arch, mode="weight_aware", weights=W, biases=b
        )
        assert score == 2

    def test_unknown_mode_raises(self):
        arch = ArchitectureSpec(layer_widths=(2, 4, 1))
        with pytest.raises(ValueError, match="Unknown mode"):
            tropical_expressivity_score(arch, mode="magic")

    def test_batch_scoring(self):
        archs = [
            ArchitectureSpec(layer_widths=(2, 4, 1)),
            ArchitectureSpec(layer_widths=(2, 8, 1)),
            ArchitectureSpec(layer_widths=(2, 4, 4, 1)),
        ]
        scores = score_batch(archs)
        assert len(scores) == 3
        assert all(s >= 1 for s in scores)

    def test_scoring_speed_generic(self):
        """Generic scoring should be well under 100ms."""
        arch = ArchitectureSpec(layer_widths=(32, 64, 64, 64, 10))
        result = benchmark_scoring_speed(arch, mode="generic", n_trials=50)
        assert result["meets_target"], (
            f"Scoring took {result['max_ms']:.1f}ms, exceeds 100ms target"
        )
        assert result["mean_ms"] < 1.0  # generic should be < 1ms


# ===================================================================
#  Evolutionary Search
# ===================================================================

class TestEvolutionarySearch:
    def test_search_converges(self):
        """Search should find architectures with improving scores."""
        space = FeedforwardSearchSpace(
            input_dim=2, output_dim=1,
            min_depth=1, max_depth=3,
            min_width=2, max_width=16,
        )
        search = EvolutionarySearch(
            search_space=space,
            population_size=20,
            n_generations=10,
            seed=42,
        )
        result = search.search()

        assert result.best_score > 0
        assert len(result.history) == 10
        assert result.total_evaluations > 0
        assert result.wall_time_seconds > 0

        # Best score should not decrease over generations
        best_scores = [h["best_score"] for h in result.history]
        for i in range(1, len(best_scores)):
            assert best_scores[i] >= best_scores[i - 1], (
                f"Best score decreased: {best_scores[i-1]} → {best_scores[i]}"
            )

    def test_search_with_nas_bench_space(self):
        """Search over NAS-Bench-201 space."""
        space = NASBench201SearchSpace(channels=8, num_classes=10)
        search = EvolutionarySearch(
            search_space=space,
            population_size=10,
            n_generations=5,
            seed=42,
        )
        result = search.search()
        assert result.best_arch.cell_ops is not None
        assert result.best_score > 0

    def test_search_caches_evaluations(self):
        """Repeated architectures should be cached, not re-evaluated."""
        space = FeedforwardSearchSpace(
            input_dim=1, output_dim=1,
            min_depth=1, max_depth=2,
            min_width=2, max_width=4,
        )
        search = EvolutionarySearch(
            search_space=space,
            population_size=20,
            n_generations=20,
            seed=42,
        )
        result = search.search()
        # With a small search space, many architectures will be cached
        assert result.history[-1]["cache_size"] <= result.total_evaluations

    def test_search_max_evaluations(self):
        """Search respects max_evaluations budget."""
        space = FeedforwardSearchSpace(input_dim=2, output_dim=1)
        search = EvolutionarySearch(
            search_space=space,
            population_size=10,
            n_generations=100,
            seed=42,
        )
        result = search.search(max_evaluations=30)
        assert result.total_evaluations <= 35  # small overshoot OK

    def test_callback(self):
        """Callback is invoked each generation."""
        space = FeedforwardSearchSpace(input_dim=2, output_dim=1)
        search = EvolutionarySearch(
            search_space=space,
            population_size=10,
            n_generations=5,
            seed=42,
        )
        calls = []
        result = search.search(callback=lambda gen, stats: calls.append(gen))
        assert calls == [0, 1, 2, 3, 4]
