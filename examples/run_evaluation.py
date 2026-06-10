import os
import glob
import time
import argparse
import pandas as pd
from typing import List

from canopy.nas.architecture import NASBench201SearchSpace
from canopy.nas.benchmark import create_benchmark, compare_metrics
from canopy.evaluation.runner import ExperimentRunner
from canopy.evaluation.visualization import (
    plot_search_trajectory,
    plot_search_trajectory_zoomed,
    plot_rank_correlation_bar,
    plot_score_vs_width_binned
)

def run_significance_test(csv_path: str):
    """Run Wilcoxon signed-rank test on the final accuracies of tropical_generic vs param_count."""
    import scipy.stats as stats
    df = pd.read_csv(csv_path)
    
    # Get final best test accuracy per seed per strategy
    idx = df.groupby(["strategy", "seed"])["evaluations"].idxmax()
    df_final = df.loc[idx]
    
    trop = df_final[df_final["strategy"] == "tropical_generic"].sort_values("seed")["best_test_accuracy"].values
    param = df_final[df_final["strategy"] == "param_count"].sort_values("seed")["best_test_accuracy"].values
    
    if len(trop) == len(param) and len(trop) >= 3:
        # Wilcoxon signed-rank test
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


def run_diagnostic_scatter(benchmark, max_archs: int, dataset_name: str, output_dir: str):
    """Generates the raw data for diagnosing the bottleneck failure and plots it."""
    from canopy.nas.tropical_score import tropical_expressivity_score
    print(f"    -> Generating diagnostic data for {dataset_name}...")
    
    entries = benchmark.get_all(max_entries=max_archs)
    rows = []
    
    for e in entries:
        score_gen = float(tropical_expressivity_score(e.arch, mode="generic"))
        score_tight = float(tropical_expressivity_score(e.arch, mode="tight"))
        rows.append({
            "arch_str": e.arch_str,
            "params": e.params,
            "test_accuracy": e.test_accuracy,
            "tropical_generic": score_gen,
            "tropical_tight": score_tight
        })
        
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f"diagnostic_data_{dataset_name}.csv")
    df.to_csv(csv_path, index=False)
    
    plot_score_vs_width_binned(csv_path, output_dir, dataset_name)


def main():
    parser = argparse.ArgumentParser(description="CANOPY Evaluation Pipeline")
    parser.add_argument("--budget", type=int, default=100, help="Max evaluations per search")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--pop-size", type=int, default=20, help="Population size")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--max-archs", type=int, default=500, help="Max architectures for correlation (saves I/O time)")
    parser.add_argument("--datasets", nargs="+", default=["cifar10"], help="Datasets to evaluate on")
    args = parser.parse_args()

    print("="*60, flush=True)
    print(" CANOPY Phase 6: Polish and Significance ", flush=True)
    print("="*60, flush=True)

    search_space = NASBench201SearchSpace()

    for ds in args.datasets:
        print(f"\n>>> Evaluating on Dataset: {ds.upper()} <<<", flush=True)
        benchmark = create_benchmark(dataset=ds)
        
        # 1. Rank Correlation Analysis
        print(f"\n  [1/3] Computing Rank Correlations on {args.max_archs} archs...", flush=True)
        comp_results = compare_metrics(benchmark, max_entries=args.max_archs)
        for r in comp_results:
            print(f"    - {r.metric_name:15s} | Kendall tau: {r.kendall_tau:.3f} | Top-10 Acc: {r.top_10_accuracy:.2f}%", flush=True)
            
        plot_rank_correlation_bar(comp_results, output_dir=args.output)
        os.replace(os.path.join(args.output, "rank_correlation.png"), os.path.join(args.output, f"rank_correlation_{ds}.png"))
        
        # 2. Diagnostic Scatter
        print(f"\n  [2/3] Diagnosing Tropical Tight Bottleneck...", flush=True)
        run_diagnostic_scatter(benchmark, args.max_archs, ds, args.output)

        # 3. End-to-End Evolutionary Search
        print(f"\n  [3/3] Running Evolutionary Search ({args.seeds} seeds)...", flush=True)
        runner = ExperimentRunner(
            search_space=search_space,
            benchmark=benchmark,
            strategies=["tropical_generic", "param_count", "random"],
            population_size=args.pop_size,
            n_generations=args.budget // args.pop_size + 1,
            max_evaluations=args.budget,
            output_dir=args.output,
        )
        
        runner.run_all(n_seeds=args.seeds)
        
        # Find the latest CSV
        csv_files = glob.glob(os.path.join(args.output, "search_trajectory_*.csv"))
        if csv_files:
            latest_csv = max(csv_files, key=os.path.getctime)
            
            # Since plots overwrite, we can move them or just rename them
            df = pd.read_csv(latest_csv)
            plot_search_trajectory(latest_csv, output_dir=args.output)
            plot_search_trajectory_zoomed(latest_csv, output_dir=args.output, max_evals=50)
            
            os.replace(os.path.join(args.output, "search_trajectory.png"), os.path.join(args.output, f"search_trajectory_{ds}.png"))
            os.replace(os.path.join(args.output, "search_trajectory_zoomed.png"), os.path.join(args.output, f"search_trajectory_zoomed_{ds}.png"))
            
            run_significance_test(latest_csv)
            
    print("\nEvaluation complete! Check the `results/` folder for plots.")


if __name__ == "__main__":
    main()
