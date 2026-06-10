"""
Evaluation runner for benchmarking expressivity-guided NAS against baselines.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

from canopy.nas.architecture import SearchSpace
from canopy.nas.search import EvolutionarySearch
from canopy.nas.benchmark import NATSBenchAPI, SyntheticBenchmark, create_benchmark


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
    ):
        self.search_space = search_space
        self.benchmark = benchmark
        self.strategies = strategies or [
            "tropical_generic",
            "tropical_tight",
            "param_count",
            "random",
        ]
        self.population_size = population_size
        self.n_generations = n_generations
        self.max_evaluations = max_evaluations
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    def _get_fitness_fn(self, strategy: str):
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
            def random_fitness(arch):
                return float(np.random.rand())
            return random_fitness

        return None  # Will default to tropical scoring

    def _get_scoring_mode(self, strategy: str) -> str:
        """Map strategy string to tropical scoring mode."""
        if strategy == "tropical_tight":
            return "tight"
        return "generic"

    def run_seed(self, seed: int) -> Dict[str, Any]:
        """Run all strategies for a single random seed."""
        results = {}

        for strategy in self.strategies:
            print(f"Running strategy: {strategy} (Seed: {seed})")
            fitness_fn = self._get_fitness_fn(strategy)
            scoring_mode = self._get_scoring_mode(strategy)

            searcher = EvolutionarySearch(
                search_space=self.search_space,
                population_size=self.population_size,
                n_generations=self.n_generations,
                scoring_mode=scoring_mode,
                seed=seed,
                fitness_fn=fitness_fn,
            )

            trajectory = []
            
            # Keep track of the highest true accuracy found so far in the population
            # (Note: standard NAS doesn't know true accuracy during search, 
            # we just track it for evaluation plotting).
            
            def callback(gen: int, stats: Dict):
                # The search stats only have the fitness score.
                # We want to know the true test accuracy of the *best found architecture* so far.
                # Actually, the callback doesn't give us the population, so we rely on the final 
                # history processing or we can extract the true accuracy later if we track all evaluated archs.
                pass

            # Since the callback doesn't have the best arch, we will just use the final population
            # but that only gives us the final state.
            # Let's run the search and extract the ground-truth accuracy trajectory manually by intercepting evaluate!

            # To properly track trajectory, we need to know the true accuracy of the arch that the search
            # thought was "best" at each evaluation step. Or simply the max true accuracy of any evaluated arch.
            # We will patch the fitness function locally to also record the true accuracy.

            eval_true_accs = []
            original_fitness_fn = fitness_fn
            
            def tracking_fitness(arch):
                entry = self.benchmark.query_by_arch(arch)
                true_acc = entry.test_accuracy if entry else 0.0
                eval_true_accs.append(true_acc)
                
                if original_fitness_fn:
                    return original_fitness_fn(arch)
                else:
                    from canopy.nas.tropical_score import tropical_expressivity_score
                    return float(tropical_expressivity_score(arch, mode=scoring_mode))

            searcher.fitness_fn = tracking_fitness

            t0 = time.perf_counter()
            search_result = searcher.search(max_evaluations=self.max_evaluations)
            wall_time = time.perf_counter() - t0

            # Compute the cumulative max of true accuracies (the best seen so far)
            cum_max_accs = np.maximum.accumulate(eval_true_accs).tolist()

            # Find true accuracy of the final chosen best architecture
            best_entry = self.benchmark.query_by_arch(search_result.best_arch)
            final_true_acc = best_entry.test_accuracy if best_entry else 0.0

            results[strategy] = {
                "final_true_accuracy": final_true_acc,
                "final_fitness_score": search_result.best_score,
                "total_evaluations": search_result.total_evaluations,
                "wall_time": wall_time,
                "trajectory_evals": list(range(1, len(cum_max_accs) + 1)),
                "trajectory_accuracy": cum_max_accs,
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
        timestamp = int(time.time())
        output_file = os.path.join(self.output_dir, f"search_results_{timestamp}.json")
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
            
        print(f"\nSaved raw results to {output_file}")
        
        # Save to CSV for pandas
        rows = []
        for res in all_results:
            seed = res["seed"]
            for strategy, stats in res["results"].items():
                for step, acc in zip(stats["trajectory_evals"], stats["trajectory_accuracy"]):
                    rows.append({
                        "seed": seed,
                        "strategy": strategy,
                        "evaluations": step,
                        "best_test_accuracy": acc,
                    })
        df = pd.DataFrame(rows)
        csv_file = os.path.join(self.output_dir, f"search_trajectory_{timestamp}.csv")
        df.to_csv(csv_file, index=False)
        print(f"Saved trajectory dataframe to {csv_file}")
        
        return all_results
