"""
Evaluation runner for benchmarking expressivity-guided NAS against baselines.
"""

import os
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

from canopy import __version__
from canopy.nas.architecture import SearchSpace
from canopy.nas.search import EvolutionarySearch
from canopy.nas.benchmark import NATSBenchAPI, SyntheticBenchmark


class ExperimentRunner:
    """
    Runs evolutionary search over multiple seeds and scoring strategies,
    tracking the ground-truth accuracy trajectories.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        benchmark,
        strategies: Optional[List[str]] = None,
        population_size: int = 50,
        n_generations: int = 50,
        max_evaluations: int = 500,
        output_dir: str = "results",
        benchmark_source_id: Optional[str] = None,
        benchmark_sha256: Optional[str] = None,
    ):
        supported = {"tropical_generic", "tropical_tight", "param_count", "random"}
        selected = list(strategies) if strategies is not None else [
            "tropical_generic", "tropical_tight", "param_count", "random"
        ]
        if not selected or any(strategy not in supported for strategy in selected):
            raise ValueError(f"strategies must be nonempty and chosen from {sorted(supported)}")
        if benchmark_sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", benchmark_sha256):
            raise ValueError("benchmark_sha256 must be a 64-character hexadecimal digest")

        self.benchmark_source_id = benchmark_source_id
        self.benchmark_sha256 = benchmark_sha256.lower() if benchmark_sha256 else None
        self.search_space = search_space
        self.benchmark = benchmark
        self.strategies = selected
        self.population_size = population_size
        self.n_generations = n_generations
        self.max_evaluations = max_evaluations
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    def _get_fitness_fn(self, strategy: str, *, seed: Optional[int] = None):
        """Return a custom fitness function for non-tropical strategies."""
        if strategy == "param_count":
            def param_fitness(arch):
                entry = self.benchmark.query_by_arch(arch)
                if entry is None:
                    return -1.0
                # Higher params = better (to simulate naive scaling)
                return float(entry.params)
            return param_fitness

        elif strategy == "random":
            # Independent per-experiment RNG. Never mutate process-global NumPy state.
            rng = np.random.default_rng(seed)

            def random_fitness(arch):
                return float(rng.random())
            return random_fitness

        return None  # Will default to tropical scoring

    def _get_scoring_mode(self, strategy: str) -> str:
        """Map strategy string to tropical scoring mode."""
        if strategy == "tropical_tight":
            return "tight"
        return "generic"


    @staticmethod
    def _score_units(strategy: str) -> str:
        return {
            "tropical_generic": "count_bound",
            "tropical_tight": "count_bound",
            "param_count": "benchmark_parameter_score",
            "random": "unit_interval_random",
        }[strategy]

    @staticmethod
    def _score_interpretation(strategy: str) -> str:
        return {
            "tropical_generic": "Architecture-level combinatorial upper bound, not exact realized regions",
            "tropical_tight": "Bottleneck-aware upper bound, not exact realized regions",
            "param_count": "Parameter count returned by the benchmark, whose units may vary",
            "random": "Seeded uniformly sampled baseline score, not a performance prediction",
        }[strategy]

    def _accuracy_source(self) -> str:
        if isinstance(self.benchmark, SyntheticBenchmark):
            return "synthetic_simulation"
        if isinstance(self.benchmark, NATSBenchAPI):
            return "nats_bench_reported"
        return "unclassified_benchmark"

    @staticmethod
    def _git_revision() -> Optional[str]:
        """Record the checked-out source revision, if git metadata is available."""
        try:
            root = Path(__file__).resolve().parents[2]
            proc = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, check=True,
                capture_output=True, text=True, timeout=3,
            )
            commit = proc.stdout.strip()
            return commit if re.fullmatch(r"[0-9a-fA-F]{40}", commit) else None
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None

    @staticmethod
    def _dependency_versions() -> Dict[str, Optional[str]]:
        packages = ("numpy", "scipy", "pandas", "nats-bench")
        versions: Dict[str, Optional[str]] = {}
        for package in packages:
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = None
        return versions

    def _benchmark_metadata(self) -> Dict[str, Any]:
        if isinstance(self.benchmark, SyntheticBenchmark):
            return {
                "kind": "synthetic",
                "accuracy_source": "synthetic_simulation",
                "dataset": "synthetic",
                "generator_seed": self.benchmark.seed,
                "generator_noise_std": self.benchmark.noise_std,
                "n_architectures": self.benchmark.n_architectures,
                "source_identifier": None,
                "source_sha256": None,
                "checksum_status": "not_applicable",
            }
        if isinstance(self.benchmark, NATSBenchAPI):
            return {
                "kind": "real_nats_bench",
                "accuracy_source": "nats_bench_reported",
                "dataset": self.benchmark.dataset,
                "hyperparameter_setting": self.benchmark.hp,
                "source_identifier": self.benchmark_source_id,
                "source_sha256": self.benchmark_sha256,
                "checksum_status": (
                    "user_supplied_not_verified"
                    if self.benchmark_sha256 else "not_recorded"
                ),
            }
        return {
            "kind": "unknown",
            "accuracy_source": "unclassified_benchmark",
            "dataset": None,
            "source_identifier": self.benchmark_source_id,
            "source_sha256": self.benchmark_sha256,
            "checksum_status": (
                "user_supplied_not_verified"
                if self.benchmark_sha256 else "not_recorded"
            ),
        }

    def _build_manifest(
        self, *, n_seeds: int, output_files: Dict[str, str],
    ) -> Dict[str, Any]:
        return {
            "manifest_schema_version": 1,
            "canopy_version": __version__,
            "git_commit": self._git_revision(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "package_versions": self._dependency_versions(),
            "benchmark": self._benchmark_metadata(),
            "missing_benchmark_data_policy": (
                "Missing benchmark lookups are represented as null; "
                "best-seen accuracy includes only observed entries."
            ),
            "configuration": {
                "search_space": type(self.search_space).__name__,
                "strategies": list(self.strategies),
                "population_size": self.population_size,
                "n_generations": self.n_generations,
                "max_evaluations": self.max_evaluations,
                "seeds": list(range(n_seeds)),
                "fitness": {
                    name: {
                        "units": self._score_units(name),
                        "interpretation": self._score_interpretation(name),
                    }
                    for name in self.strategies
                },
            },
            "files": output_files,
        }

    def run_seed(self, seed: int) -> Dict[str, Any]:
        """Run all strategies for a single random seed."""
        results = {}

        for strategy in self.strategies:
            print(f"Running strategy: {strategy} (Seed: {seed})")
            fitness_fn = self._get_fitness_fn(strategy, seed=seed)
            scoring_mode = self._get_scoring_mode(strategy)

            searcher = EvolutionarySearch(
                search_space=self.search_space,
                population_size=self.population_size,
                n_generations=self.n_generations,
                scoring_mode=scoring_mode,
                seed=seed,
                fitness_fn=fitness_fn,
            )

            # Track the highest reported accuracy encountered for each newly
            # evaluated architecture. The searcher caches duplicate candidates.
            eval_true_accs = []
            original_fitness_fn = fitness_fn
            
            def tracking_fitness(arch):
                entry = self.benchmark.query_by_arch(arch)
                true_acc = float(entry.test_accuracy) if entry is not None else None
                eval_true_accs.append(true_acc)
                
                if original_fitness_fn:
                    return original_fitness_fn(arch)
                else:
                    from canopy.nas.tropical_score import tropical_expressivity_score
                    return float(tropical_expressivity_score(arch, mode=scoring_mode))

            searcher.fitness_fn = tracking_fitness

            search_result = searcher.search(max_evaluations=self.max_evaluations)
            wall_time = search_result.wall_time_seconds

            # Missing benchmark observations must remain missing, not be
            # silently turned into measured 0% accuracies.
            cum_max_accs = []
            best_seen = None
            for value in eval_true_accs:
                if value is not None:
                    best_seen = value if best_seen is None else max(best_seen, value)
                cum_max_accs.append(best_seen)

            best_entry = self.benchmark.query_by_arch(search_result.best_arch)
            final_true_acc = (
                float(best_entry.test_accuracy) if best_entry is not None else None
            )

            results[strategy] = {
                "final_true_accuracy": final_true_acc,
                "final_fitness_score": search_result.best_score,
                "score_units": self._score_units(strategy),
                "score_interpretation": self._score_interpretation(strategy),
                "total_evaluations": search_result.total_evaluations,
                "wall_time": wall_time,
                "trajectory_evals": list(range(1, len(cum_max_accs) + 1)),
                "trajectory_accuracy": cum_max_accs,
                "trajectory_lookup_available": [
                    value is not None for value in eval_true_accs
                ],
                "missing_benchmark_lookups": sum(
                    value is None for value in eval_true_accs
                ),
            }

        return results

    def run_all(self, n_seeds: int = 5) -> List[Dict]:
        """Run the evaluation over multiple seeds and aggregate results."""
        all_results = []
        for seed in range(n_seeds):
            print(f"--- Starting Seed {seed} ---")
            seed_results = self.run_seed(seed)
            all_results.append({"seed": seed, "results": seed_results})

        # Save to JSON
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_file = os.path.join(self.output_dir, f"search_results_{timestamp}.json")
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
            
        print(f"\nSaved raw results to {output_file}")
        
        # Save to CSV for pandas
        rows = []
        for res in all_results:
            seed = res["seed"]
            for strategy, stats in res["results"].items():
                for step, acc, observed in zip(
                    stats["trajectory_evals"],
                    stats["trajectory_accuracy"],
                    stats["trajectory_lookup_available"],
                ):
                    rows.append({
                        "seed": seed,
                        "strategy": strategy,
                        "evaluations": step,
                        "best_test_accuracy": acc,
                        "accuracy_source": self._accuracy_source(),
                        "benchmark_lookup_available": observed,
                        "fitness_score_units": stats["score_units"],
                    })
        df = pd.DataFrame(rows)
        csv_file = os.path.join(self.output_dir, f"search_trajectory_{timestamp}.csv")
        df.to_csv(csv_file, index=False)
        print(f"Saved trajectory dataframe to {csv_file}")

        manifest_file = os.path.join(self.output_dir, f"experiment_manifest_{timestamp}.json")
        manifest = self._build_manifest(
            n_seeds=n_seeds,
            output_files={
                "results_json": os.path.basename(output_file),
                "trajectory_csv": os.path.basename(csv_file),
            },
        )
        with open(manifest_file, "w", encoding="utf-8") as output:
            json.dump(manifest, output, indent=2, sort_keys=True)
        print(f"Saved reproducibility manifest to {manifest_file}")

        return all_results
