import os
import glob
import time
import argparse
import pandas as pd
import numpy as np
import logging
from typing import List
from dataclasses import dataclass

from canopy.nas.architecture import DARTS_SearchSpace, ArchitectureSpec
from canopy.evaluation.runner import ExperimentRunner
from canopy.evaluation.visualization import (
    plot_search_trajectory,
    plot_search_trajectory_zoomed,
)

logger = logging.getLogger(__name__)

@dataclass
class MockEntry:
    test_accuracy: float
    params: float

class DARTSBenchmarkWrapper:
    def __init__(self):
        import nasbench301 as nb
        print("Loading NAS-Bench-301 Surrogate Models (1.8GB)...")
        models_1_0_dir = os.path.expanduser("~/nb301_models")
        self.performance_model = nb.load_ensemble(models_1_0_dir)
        print("Models loaded.")

    def query_by_arch(self, arch: ArchitectureSpec) -> MockEntry:
        from nasbench301.representations import Genotype
        from nasbench301.api import fixed_hyperparameters

        normal = []
        p_count = 0
        for op_entry in arch.cell_ops:
            op, src = op_entry.split("~")
            normal.append((op, int(src)))
            if "conv_3x3" in op: p_count += 5
            elif "conv_5x5" in op: p_count += 9
            elif "pool" in op: p_count += 1
            
        genotype = Genotype(normal=normal, normal_concat=[2,3,4,5], reduce=normal, reduce_concat=[2,3,4,5])
        config_dict = self.performance_model.convert_representation(genotype, "genotype")
        
        full_config = {**fixed_hyperparameters, **config_dict}
        full_config['NetworkSelectorDatasetInfo:darts:init_channels'] = 50
        full_config['NetworkSelectorDatasetInfo:darts:layers'] = 20
        
        try:
            acc = self.performance_model.model.query(full_config)
        except Exception as e:
            raise RuntimeError(
                "NAS-Bench-301 surrogate query failed; cannot substitute "
                "a fixed value for actual accuracy."
            ) from e
            
        return MockEntry(test_accuracy=acc, params=p_count)

def run_significance_test(csv_path: str):
    """Run Wilcoxon signed-rank test on the final accuracies of tropical_generic vs param_count."""
    import scipy.stats as stats
    df = pd.read_csv(csv_path)
    
    idx = df.groupby(["strategy", "seed"])["evaluations"].idxmax()
    df_final = df.loc[idx]
    
    trop = df_final[df_final["strategy"] == "tropical_generic"].sort_values("seed")["best_test_accuracy"].values
    param = df_final[df_final["strategy"] == "param_count"].sort_values("seed")["best_test_accuracy"].values
    
    if len(trop) == len(param) and len(trop) >= 3:
        stat, p = stats.wilcoxon(trop, param, alternative='greater')
        margin = (trop.mean() - param.mean())
        print(f"\n  [SIGNIFICANCE TEST] Tropical Generic vs Param Count:")
        print(f"    - Margin (Mean Gap): +{margin:.3f}%")
        print(f"    - Wilcoxon p-value:  {p:.4f}")
        if p < 0.05:
            print(f"    - Conclusion: STATISTICALLY SIGNIFICANT (p < 0.05) ✅")
        else:
            print(f"    - Conclusion: NOT SIGNIFICANT (p >= 0.05) ❌")
    else:
        print("\n  [SIGNIFICANCE TEST] Not enough data or mismatched seeds to run Wilcoxon.")

def configure_darts_fitness(runner):
    """Use the DARTS proxy while preserving the runner's seeded fitness API."""
    original_get_fitness = runner._get_fitness_fn

    def custom_get_fitness(strategy, *, seed=None):
        if strategy == "tropical_generic":
            from canopy.nas.tropical_score import tropical_expressivity_score_darts
            return lambda arch: float(tropical_expressivity_score_darts(arch))
        return original_get_fitness(strategy, seed=seed)

    runner._get_fitness_fn = custom_get_fitness


def main():
    parser = argparse.ArgumentParser(description="CANOPY Evaluation Pipeline - DARTS Search Trajectories")
    parser.add_argument("--budget", type=int, default=150, help="Max evaluations per search")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--pop-size", type=int, default=20, help="Population size")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    print("="*60, flush=True)
    print(" CANOPY Phase 9: NAS-Bench-301 Search Trajectory Evaluation ", flush=True)
    print("="*60, flush=True)

    search_space = DARTS_SearchSpace()
    benchmark = DARTSBenchmarkWrapper()

    runner = ExperimentRunner(
        search_space=search_space,
        benchmark=benchmark,
        strategies=["tropical_generic", "param_count", "random"],
        population_size=args.pop_size,
        n_generations=args.budget // args.pop_size + 1,
        max_evaluations=args.budget,
        output_dir=args.output,
    )
    
    # Preserve per-seed random baselines when overriding the DARTS score.
    configure_darts_fitness(runner)
    
    runner.run_all(n_seeds=args.seeds)
    
    csv_files = glob.glob(os.path.join(args.output, "search_trajectory_*.csv"))
    if csv_files:
        latest_csv = max(csv_files, key=os.path.getctime)
        plot_search_trajectory(latest_csv, output_dir=args.output)
        
        # Rename plots for DARTS specifically
        os.replace(os.path.join(args.output, "search_trajectory.png"), os.path.join(args.output, "darts_search_trajectory.png"))
        
        run_significance_test(latest_csv)
        
if __name__ == "__main__":
    main()
