"""
Tests for canopy.nas.benchmark
================================

Verifies:
1. Architecture string conversion (cell_ops ↔ NAS-Bench-201 format)
2. Kendall τ and Spearman ρ implementation
3. Top-K accuracy
4. SyntheticBenchmark generation and query
5. NATSBenchAPI integration (skipped if data unavailable)
6. compare_metrics produces correct structure
7. Tropical expressivity outperforms random baseline
"""

import os
import numpy as np
import pytest

from canopy.nas.architecture import NASBench201SearchSpace
from canopy.nas.benchmark import (
    BenchmarkEntry,
    NATSBenchAPI,
    SyntheticBenchmark,
    cell_ops_to_arch_str,
    arch_str_to_cell_ops,
    create_benchmark,
    kendall_tau,
    spearman_rho,
    top_k_accuracy,
    compare_metrics,
)


# ===================================================================
#  Architecture string conversion
# ===================================================================

class TestArchStringConversion:
    def test_ops_to_str(self):
        ops = ("nor_conv_3x3", "nor_conv_1x1", "skip_connect",
               "none", "avg_pool_3x3", "nor_conv_3x3")
        s = cell_ops_to_arch_str(ops)
        assert "nor_conv_3x3~0" in s
        assert "nor_conv_1x1~0" in s
        assert "skip_connect~1" in s
        assert "none~0" in s
        assert "avg_pool_3x3~1" in s
        assert "nor_conv_3x3~2" in s

    def test_roundtrip(self):
        """cell_ops → string → cell_ops should be identity."""
        ops = ("nor_conv_3x3", "nor_conv_1x1", "skip_connect",
               "none", "avg_pool_3x3", "nor_conv_3x3")
        s = cell_ops_to_arch_str(ops)
        recovered = arch_str_to_cell_ops(s)
        assert recovered == ops

    def test_all_same_op(self):
        ops = ("none",) * 6
        s = cell_ops_to_arch_str(ops)
        recovered = arch_str_to_cell_ops(s)
        assert recovered == ops

    def test_all_ops_represented(self):
        """Each of the 5 ops should survive roundtrip."""
        for op in NASBench201SearchSpace.OPS:
            ops = (op,) * 6
            s = cell_ops_to_arch_str(ops)
            recovered = arch_str_to_cell_ops(s)
            assert recovered == ops, f"Failed for op={op}"


# ===================================================================
#  Rank correlation functions
# ===================================================================

class TestKendallTau:
    def test_perfect_concordance(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([10.0, 20.0, 30.0, 40.0])
        assert kendall_tau(a, b) == pytest.approx(1.0)

    def test_perfect_discordance(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([40.0, 30.0, 20.0, 10.0])
        assert kendall_tau(a, b) == pytest.approx(-1.0)

    def test_no_correlation(self):
        rng = np.random.default_rng(42)
        a = rng.permutation(100).astype(float)
        b = rng.permutation(100).astype(float)
        tau = kendall_tau(a, b)
        assert -0.3 < tau < 0.3

    def test_single_element(self):
        assert kendall_tau(np.array([1.0]), np.array([2.0])) == 1.0


class TestSpearmanRho:
    def test_perfect_concordance(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([10.0, 20.0, 30.0, 40.0])
        assert spearman_rho(a, b) == pytest.approx(1.0)

    def test_perfect_discordance(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([40.0, 30.0, 20.0, 10.0])
        assert spearman_rho(a, b) == pytest.approx(-1.0)


class TestTopKAccuracy:
    def test_perfect_ranking(self):
        pred = np.array([1, 2, 3, 4, 5], dtype=float)
        true = np.array([1, 2, 3, 4, 5], dtype=float)
        assert top_k_accuracy(pred, true, k=2) == 1.0

    def test_reversed_ranking(self):
        pred = np.array([5, 4, 3, 2, 1], dtype=float)
        true = np.array([1, 2, 3, 4, 5], dtype=float)
        assert top_k_accuracy(pred, true, k=2) == 0.0

    def test_partial_overlap(self):
        pred = np.array([1, 5, 4, 2, 3], dtype=float)
        true = np.array([1, 2, 3, 4, 5], dtype=float)
        acc = top_k_accuracy(pred, true, k=2)
        assert 0.0 <= acc <= 1.0


# ===================================================================
#  Synthetic Benchmark
# ===================================================================

class TestSyntheticBenchmark:
    def test_creation(self):
        bench = SyntheticBenchmark(n_architectures=50, seed=42)
        entries = bench.get_all()
        assert len(entries) == 50
        for e in entries:
            assert 0 <= e.test_accuracy <= 100
            assert e.params > 0

    def test_best_accuracy(self):
        bench = SyntheticBenchmark(n_architectures=100, seed=42)
        assert bench.best_accuracy > 50

    def test_query_by_arch(self):
        bench = SyntheticBenchmark(n_architectures=20, seed=42)
        entries = bench.get_all()
        result = bench.query_by_arch(entries[0].arch)
        assert result is not None
        assert result.test_accuracy == entries[0].test_accuracy

    def test_query_unknown_arch(self):
        bench = SyntheticBenchmark(n_architectures=10, seed=42)
        unknown = bench.space.index_to_arch(15000)
        assert bench.query_by_arch(unknown) is None

    def test_entries_have_arch_str(self):
        bench = SyntheticBenchmark(n_architectures=10, seed=42)
        for entry in bench.get_all():
            assert entry.arch_str != ""
            assert "~" in entry.arch_str


# ===================================================================
#  NATSBenchAPI (integration — skipped if data unavailable)
# ===================================================================

_NATS_DATA_PATH = os.environ.get("NATS_BENCH_TSS_PATH", None)
_HAS_NATS_DATA = _NATS_DATA_PATH is not None and os.path.exists(
    _NATS_DATA_PATH
)


@pytest.mark.skipif(
    not _HAS_NATS_DATA,
    reason="NATS-Bench data not found. Set NATS_BENCH_TSS_PATH env var."
)
class TestNATSBenchAPI:
    @pytest.fixture(scope="class", autouse=True)
    def setup(self, request):
        request.cls.api = NATSBenchAPI(_NATS_DATA_PATH, dataset="cifar10", hp="200")

    def test_total_architectures(self):
        assert self.api.total_architectures == 15625

    def test_query_by_index(self):
        entry = self.api.query_by_index(0)
        assert isinstance(entry, BenchmarkEntry)
        assert 0 <= entry.test_accuracy <= 100
        assert entry.index == 0
        assert entry.arch.cell_ops is not None

    def test_query_by_index_cached(self):
        e1 = self.api.query_by_index(42)
        e2 = self.api.query_by_index(42)
        assert e1.test_accuracy == e2.test_accuracy

    def test_query_by_arch(self):
        entry = self.api.query_by_index(100)
        result = self.api.query_by_arch(entry.arch)
        assert result is not None
        assert result.test_accuracy == entry.test_accuracy

    def test_get_all_subset(self):
        entries = self.api.get_all(max_entries=10)
        assert len(entries) == 10
        for e in entries:
            assert isinstance(e, BenchmarkEntry)

    def test_arch_str_format(self):
        entry = self.api.query_by_index(0)
        assert "|" in entry.arch_str
        assert "~" in entry.arch_str


# ===================================================================
#  create_benchmark
# ===================================================================

class TestCreateBenchmark:
    def test_synthetic_fallback(self, monkeypatch):
        monkeypatch.delenv("NATS_BENCH_TSS_PATH", raising=False)
        bench = create_benchmark(data_path=None, synthetic_fallback=True)
        assert isinstance(bench, SyntheticBenchmark)

    def test_default_requires_real_benchmark_data(self, monkeypatch):
        """Missing real data must not silently produce synthetic correlations."""
        monkeypatch.delenv("NATS_BENCH_TSS_PATH", raising=False)
        with pytest.raises(FileNotFoundError, match="NATS-Bench"):
            create_benchmark()

    def test_no_fallback_raises(self):
        with pytest.raises(FileNotFoundError, match="NATS-Bench"):
            create_benchmark(
                data_path="/nonexistent/path.pth",
                synthetic_fallback=False,
            )

    @pytest.mark.skipif(
        not _HAS_NATS_DATA,
        reason="NATS-Bench data not found."
    )
    def test_real_data(self):
        bench = create_benchmark(data_path=_NATS_DATA_PATH)
        assert isinstance(bench, NATSBenchAPI)


# ===================================================================
#  compare_metrics
# ===================================================================

class TestCompareMetrics:
    def test_produces_all_baselines(self):
        bench = SyntheticBenchmark(n_architectures=100, seed=42)
        results = compare_metrics(bench, seed=42)
        names = [r.metric_name for r in results]
        assert "tropical_generic" in names
        assert "tropical_tight" in names
        assert "param_count" in names
        assert "random" in names

    def test_tropical_beats_random(self):
        bench = SyntheticBenchmark(n_architectures=200, seed=42)
        results = compare_metrics(bench, seed=42)

        tropical = next(r for r in results if r.metric_name == "tropical_generic")
        random = next(r for r in results if r.metric_name == "random")

        assert tropical.kendall_tau > random.kendall_tau, (
            f"Tropical τ={tropical.kendall_tau:.3f} should beat "
            f"random τ={random.kendall_tau:.3f}"
        )

    def test_correlation_bounds(self):
        bench = SyntheticBenchmark(n_architectures=100, seed=42)
        results = compare_metrics(bench, seed=42)
        for r in results:
            assert -1.0 <= r.kendall_tau <= 1.0
            assert -1.0 <= r.spearman_rho <= 1.0
            assert 0.0 <= r.top_10_accuracy <= 1.0
            assert 0.0 <= r.top_50_accuracy <= 1.0

    @pytest.mark.skipif(
        not _HAS_NATS_DATA,
        reason="NATS-Bench data not found."
    )
    def test_compare_on_real_data(self):
        """Run comparison on real NAS-Bench-201 data (first 200 archs)."""
        bench = NATSBenchAPI(_NATS_DATA_PATH)
        # Limit to 200 for speed
        bench_limited = type('LimitedBench', (), {
            'get_all': lambda self: bench.get_all(max_entries=200)
        })()
        results = compare_metrics(bench_limited, seed=42)
        for r in results:
            print(r)
