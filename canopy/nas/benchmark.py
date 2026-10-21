"""
NAS Benchmark Integration and Baseline Comparison
===================================================

Wraps the real NATS-Bench / NAS-Bench-201 API for ground-truth
evaluation of architectures, with comparison against tropical
expressivity and baselines.

When the NATS-Bench data file is available, NATSBenchAPI wraps the
official `nats_bench` package.  Otherwise, a SyntheticBenchmark
provides a test-only fallback.

Baselines implemented
---------------------
- Random search
- Parameter count
- FLOPs estimate
- Tropical expressivity (generic, tight)

Metrics
-------
- Kendall τ rank correlation
- Spearman ρ rank correlation
- Top-K accuracy (does our ranking include the true top-K?)
"""

from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from canopy.nas.architecture import (
    ArchitectureSpec,
    NASBench201SearchSpace,
)
from canopy.nas.tropical_score import tropical_expressivity_score


# ---------------------------------------------------------------------------
#  NAS-Bench-201 architecture string conversion
# ---------------------------------------------------------------------------

# The 6 edges in NAS-Bench-201 correspond to:
#   edge 0: (0→1)
#   edge 1: (0→2)
#   edge 2: (1→2)
#   edge 3: (0→3)
#   edge 4: (1→3)
#   edge 5: (2→3)
# The architecture string format is:
#   |op~src|+|op~src|op~src|+|op~src|op~src|op~src|

def cell_ops_to_arch_str(cell_ops: Tuple[str, ...]) -> str:
    """
    Convert CANOPY cell_ops to NAS-Bench-201 architecture string.

    cell_ops order: (0→1), (0→2), (1→2), (0→3), (1→3), (2→3)

    The NAS-Bench-201 string format groups by target node:
      node 1: |op~0|
      node 2: |op~0|op~1|
      node 3: |op~0|op~1|op~2|
    """
    # Group edges by target node
    # edge 0: (0→1) → node 1 group
    # edge 1: (0→2), edge 2: (1→2) → node 2 group
    # edge 3: (0→3), edge 4: (1→3), edge 5: (2→3) → node 3 group
    return (
        f"|{cell_ops[0]}~0|"
        f"+|{cell_ops[1]}~0|{cell_ops[2]}~1|"
        f"+|{cell_ops[3]}~0|{cell_ops[4]}~1|{cell_ops[5]}~2|"
    )


def arch_str_to_cell_ops(arch_str: str) -> Tuple[str, ...]:
    """
    Convert NAS-Bench-201 architecture string to CANOPY cell_ops.

    Parses: |op~src|+|op~src|op~src|+|op~src|op~src|op~src|
    """
    # Split by '+' to get node groups
    groups = arch_str.split("+")
    ops: List[str] = []

    for group in groups:
        # Parse |op~src| tokens
        tokens = group.strip().split("|")
        for token in tokens:
            token = token.strip()
            if "~" in token:
                op = token.split("~")[0]
                ops.append(op)

    return tuple(ops)


# ---------------------------------------------------------------------------
#  Benchmark Entry
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkEntry:
    """A single benchmark entry with architecture and ground-truth metrics."""
    arch: ArchitectureSpec
    test_accuracy: float       # ground-truth test accuracy (0..100)
    valid_accuracy: float      # validation accuracy
    params: float              # parameter count (millions)
    flops: float               # estimated FLOPs (millions)
    index: int                 # index in the benchmark
    arch_str: str = ""         # NAS-Bench-201 architecture string
    dataset: str = "cifar10"   # dataset used


# ---------------------------------------------------------------------------
#  Real NATS-Bench API wrapper
# ---------------------------------------------------------------------------

class NATSBenchAPI:
    """
    Wrapper around the official nats_bench package for NAS-Bench-201.

    Requires:
    1. `pip install nats_bench`
    2. The NATS-Bench TSS data file downloaded from
       https://github.com/D-X-Y/NATS-Bench

    Parameters
    ----------
    data_path : str
        Path to the NATS-Bench TSS data file or directory.
    dataset : str
        Dataset to evaluate on: 'cifar10', 'cifar100', 'ImageNet16-120'.
    hp : str
        Hyperparameter setting: '12' (12 epochs) or '200' (200 epochs).
    """

    def __init__(
        self,
        data_path: str,
        dataset: str = "cifar10",
        hp: str = "200",
    ):
        import nats_bench
        # fast_mode=True requires the unpacked -simple directory.
        # For .pickle.pbz2 files, fast_mode must be False.
        is_dir = os.path.isdir(data_path)
        self._api = nats_bench.create(
            data_path, "tss", fast_mode=is_dir, verbose=False
        )
        self.dataset = dataset
        self.hp = hp
        self._space = NASBench201SearchSpace()
        self._cache: Dict[int, BenchmarkEntry] = {}

    @property
    def total_architectures(self) -> int:
        return 15625

    def query_by_index(self, index: int) -> BenchmarkEntry:
        """Look up architecture by its NAS-Bench-201 index."""
        if index in self._cache:
            return self._cache[index]

        arch_str = self._api.arch(index)
        cell_ops = arch_str_to_cell_ops(arch_str)
        arch = self._space._ops_to_arch(cell_ops)
        # Override cell_ops to match the parsed ones
        arch = ArchitectureSpec(
            layer_widths=arch.layer_widths,
            cell_ops=cell_ops,
        )

        info = self._api.get_more_info(
            index, self.dataset, hp=self.hp, is_random=False
        )
        cost = self._api.get_cost_info(index, self.dataset, hp=self.hp)

        entry = BenchmarkEntry(
            arch=arch,
            test_accuracy=info.get("test-accuracy", 0.0),
            valid_accuracy=info.get("valid-accuracy", 0.0),
            params=cost.get("params", 0.0),
            flops=cost.get("flops", 0.0),
            index=index,
            arch_str=arch_str,
            dataset=self.dataset,
        )
        self._cache[index] = entry
        return entry

    def query_by_arch(self, arch: ArchitectureSpec) -> Optional[BenchmarkEntry]:
        """Look up architecture by its specification."""
        if arch.cell_ops is None:
            return None
        arch_str = cell_ops_to_arch_str(arch.cell_ops)
        try:
            index = self._api.query_index_by_arch(arch_str)
            if index < 0:
                return None
            return self.query_by_index(index)
        except Exception:
            return None

    def get_all(
        self, *, max_entries: Optional[int] = None
    ) -> List[BenchmarkEntry]:
        """Load all (or first N) benchmark entries."""
        n = max_entries if max_entries else self.total_architectures
        n = min(n, self.total_architectures)
        entries = []
        for i in range(n):
            try:
                entries.append(self.query_by_index(i))
            except Exception:
                continue
        return entries

    def get_best(self) -> BenchmarkEntry:
        """Find the architecture with highest test accuracy."""
        best_info = self._api.find_best(
            self.dataset, metric_on_set="test", hp=self.hp
        )
        best_index = best_info if isinstance(best_info, int) else 0
        return self.query_by_index(best_index)


# ---------------------------------------------------------------------------
#  Synthetic Benchmark (fallback for testing)
# ---------------------------------------------------------------------------

class SyntheticBenchmark:
    """
    Synthetic NAS-Bench-201-like benchmark for testing when real data
    is unavailable.

    Generates architectures with synthetic ground-truth accuracies
    correlated with tropical expressivity.

    Parameters
    ----------
    search_space : NASBench201SearchSpace
    n_architectures : int
    noise_std : float
    seed : int
    """

    def __init__(
        self,
        search_space: Optional[NASBench201SearchSpace] = None,
        n_architectures: int = 500,
        noise_std: float = 2.0,
        seed: int = 42,
    ):
        self.space = search_space or NASBench201SearchSpace()
        self.n_architectures = min(n_architectures, self.space.TOTAL_ARCHITECTURES)
        self.noise_std = noise_std
        self.seed = seed
        self.entries: List[BenchmarkEntry] = []
        self._build()

    def _build(self):
        """Generate synthetic benchmark data."""
        rng = np.random.default_rng(self.seed)

        indices = rng.choice(
            self.space.TOTAL_ARCHITECTURES,
            size=self.n_architectures,
            replace=False,
        )

        # Collect all scores first, then normalize
        scores = []
        arch_data = []
        for idx in indices:
            arch = self.space.index_to_arch(int(idx))
            score = tropical_expressivity_score(arch, mode="generic")
            scores.append(float(score))
            arch_data.append((arch, int(idx)))

        scores_arr = np.array(scores, dtype=float)
        # Use log-log scaling: scores span 10^7..10^24, log gives 16..56,
        # log-log gives 2.8..4.0 — a manageable range.
        log_scores = np.log1p(np.log1p(scores_arr))
        # Normalize to [60, 95] range + noise
        s_min, s_max = log_scores.min(), log_scores.max()
        if s_max - s_min > 1e-12:
            normed = (log_scores - s_min) / (s_max - s_min)
        else:
            normed = np.full_like(log_scores, 0.5)
        accuracies = 60.0 + 35.0 * normed + rng.normal(0, self.noise_std, size=len(normed))
        accuracies = np.clip(accuracies, 0, 100)

        for i, (arch, idx) in enumerate(arch_data):
            params = arch.total_params
            flops = params * 2

            self.entries.append(BenchmarkEntry(
                arch=arch,
                test_accuracy=float(accuracies[i]),
                valid_accuracy=float(accuracies[i] - rng.uniform(0, 2)),
                params=float(params),
                flops=float(flops),
                index=idx,
                arch_str=cell_ops_to_arch_str(arch.cell_ops) if arch.cell_ops else "",
                dataset="synthetic",
            ))

    def query_by_arch(self, arch: ArchitectureSpec) -> Optional[BenchmarkEntry]:
        """Look up test accuracy for an architecture."""
        for entry in self.entries:
            if entry.arch == arch:
                return entry
        return None

    def get_all(self, **kwargs) -> List[BenchmarkEntry]:
        return self.entries

    @property
    def best_accuracy(self) -> float:
        return max(e.test_accuracy for e in self.entries)

    @property
    def best_arch(self) -> ArchitectureSpec:
        return max(self.entries, key=lambda e: e.test_accuracy).arch


# ---------------------------------------------------------------------------
#  Auto-detect benchmark
# ---------------------------------------------------------------------------

def create_benchmark(
    data_path: Optional[str] = None,
    dataset: str = "cifar10",
    hp: str = "200",
    *,
    synthetic_fallback: bool = False,
    n_synthetic: int = 500,
    seed: int = 42,
):
    """
    Create a benchmark instance.

    Attempts to use real NATS-Bench data if `data_path` is provided
    and the file exists. Requires real data by default; synthetic
    test data are used only with explicit synthetic_fallback=True.

    Parameters
    ----------
    data_path : str, optional
        Path to NATS-Bench TSS data file.  Also checks the environment
        variable NATS_BENCH_TSS_PATH.
    dataset : str
    hp : str
    synthetic_fallback : bool
        Explicit test-only opt-in to synthetic data if real data are unavailable.
        Defaults to False to prevent synthetic accuracy from being
        mistaken for real benchmark results.
    n_synthetic : int
        Number of architectures for the synthetic benchmark.
    seed : int

    Returns
    -------
    NATSBenchAPI or SyntheticBenchmark
    """
    if data_path is None:
        data_path = os.environ.get("NATS_BENCH_TSS_PATH", None)

    if data_path and os.path.exists(data_path):
        try:
            return NATSBenchAPI(data_path, dataset=dataset, hp=hp)
        except Exception as e:
            if not synthetic_fallback:
                raise
            print(f"Warning: Failed to load NATS-Bench from {data_path}: {e}")
            print("Falling back to synthetic benchmark.")

    if synthetic_fallback:
        return SyntheticBenchmark(
            n_architectures=n_synthetic, seed=seed
        )

    raise FileNotFoundError(
        f"NATS-Bench data not found at {data_path!r}. "
        f"Set NATS_BENCH_TSS_PATH or pass data_path explicitly. "
        f"Download from https://github.com/D-X-Y/NATS-Bench"
    )


# ---------------------------------------------------------------------------
#  Scoring baselines
# ---------------------------------------------------------------------------

def _param_count_score(arch: ArchitectureSpec) -> float:
    return float(arch.total_params)


def _flops_score(arch: ArchitectureSpec) -> float:
    return float(arch.total_params * 2)


def _random_score(arch: ArchitectureSpec, rng: np.random.Generator) -> float:
    return float(rng.random())


# ---------------------------------------------------------------------------
#  Ranking comparison
# ---------------------------------------------------------------------------

def kendall_tau(ranking_a: np.ndarray, ranking_b: np.ndarray) -> float:
    """
    Kendall's τ rank correlation coefficient.

    τ = (concordant - discordant) / (n*(n-1)/2)
    """
    n = len(ranking_a)
    if n < 2:
        return 1.0

    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_diff = ranking_a[i] - ranking_a[j]
            b_diff = ranking_b[i] - ranking_b[j]
            product = a_diff * b_diff
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1

    denom = n * (n - 1) / 2
    return (concordant - discordant) / denom if denom > 0 else 0.0


def spearman_rho(ranking_a: np.ndarray, ranking_b: np.ndarray) -> float:
    """Spearman's ρ via Pearson correlation on ranks."""
    from scipy.stats import rankdata
    r_a = rankdata(ranking_a)
    r_b = rankdata(ranking_b)
    n = len(r_a)
    if n < 2:
        return 1.0
    mean_a, mean_b = r_a.mean(), r_b.mean()
    cov = np.sum((r_a - mean_a) * (r_b - mean_b))
    std_a = np.sqrt(np.sum((r_a - mean_a) ** 2))
    std_b = np.sqrt(np.sum((r_b - mean_b) ** 2))
    if std_a < 1e-12 or std_b < 1e-12:
        return 0.0
    return float(cov / (std_a * std_b))


def top_k_accuracy(
    predicted_scores: np.ndarray,
    true_scores: np.ndarray,
    k: int = 10,
) -> float:
    """Fraction of the true top-K found in the predicted top-K."""
    n = len(predicted_scores)
    k = min(k, n)
    true_top_k = set(np.argsort(true_scores)[-k:])
    pred_top_k = set(np.argsort(predicted_scores)[-k:])
    return len(true_top_k & pred_top_k) / k


@dataclass
class ComparisonResult:
    """Result of comparing a NAS metric against ground truth."""
    metric_name: str
    kendall_tau: float
    spearman_rho: float
    top_10_accuracy: float
    top_50_accuracy: float

    def __repr__(self) -> str:
        return (
            f"{self.metric_name}: τ={self.kendall_tau:.3f}, "
            f"ρ={self.spearman_rho:.3f}, "
            f"top10={self.top_10_accuracy:.1%}, "
            f"top50={self.top_50_accuracy:.1%}"
        )


def compare_metrics(
    benchmark,
    *,
    seed: int = 42,
    max_entries: Optional[int] = None,
) -> List[ComparisonResult]:
    """
    Compare tropical NAS metrics against baselines.

    Works with both NATSBenchAPI and SyntheticBenchmark — any object
    with a .get_all() method returning List[BenchmarkEntry].

    Returns a list of ComparisonResult, one per metric.
    """
    entries = benchmark.get_all(max_entries=max_entries)
    true_acc = np.array([e.test_accuracy for e in entries])
    archs = [e.arch for e in entries]

    rng = np.random.default_rng(seed)

    metrics: Dict[str, np.ndarray] = {
        "tropical_generic": np.array([
            tropical_expressivity_score(a, mode="generic") for a in archs
        ], dtype=float),
        "tropical_tight": np.array([
            tropical_expressivity_score(a, mode="tight") for a in archs
        ], dtype=float),
        "param_count": np.array([
            _param_count_score(a) for a in archs
        ]),
        "flops": np.array([
            _flops_score(a) for a in archs
        ]),
        "random": np.array([
            _random_score(a, rng) for a in archs
        ]),
    }

    results = []
    for name, scores in metrics.items():
        results.append(ComparisonResult(
            metric_name=name,
            kendall_tau=kendall_tau(scores, true_acc),
            spearman_rho=spearman_rho(scores, true_acc),
            top_10_accuracy=top_k_accuracy(scores, true_acc, k=10),
            top_50_accuracy=top_k_accuracy(scores, true_acc, k=50),
        ))

    return results
