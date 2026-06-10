"""
Visualization tools for evaluating NAS strategies.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from canopy.nas.benchmark import ComparisonResult

# Set academic plotting style
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})


def plot_search_trajectory(csv_path: str, output_dir: str = "results"):
    """
    Plot the search trajectory (best test accuracy vs. evaluations).
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 6))
    
    # Plot lineplot with confidence intervals across seeds
    sns.lineplot(
        data=df, 
        x="evaluations", 
        y="best_test_accuracy", 
        hue="strategy",
        errorbar=("ci", 95), # 95% confidence interval
        linewidth=2.5
    )

    plt.title("Evolutionary Search Convergence: Tropical vs Baselines")
    plt.xlabel("Number of Evaluated Architectures")
    plt.ylabel("Best Test Accuracy Found (%)")
    plt.legend(title="Fitness Strategy", loc="lower right")
    
    out_file = os.path.join(output_dir, "search_trajectory.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved trajectory plot to {out_file}")


def plot_score_vs_accuracy_scatter(
    benchmark_results: List[ComparisonResult], 
    output_dir: str = "results"
):
    """
    Create a scatter plot of metric score vs. ground truth accuracy
    for each baseline in the comparison results.
    """
    os.makedirs(output_dir, exist_ok=True)

    n_metrics = len(benchmark_results)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), sharey=True)
    
    if n_metrics == 1:
        axes = [axes]

    for i, res in enumerate(benchmark_results):
        ax = axes[i]
        
        # We need the raw data for scatter plots. 
        # But wait, ComparisonResult only has aggregate metrics!
        # It doesn't store the raw pairs. Let's make sure we have access to it or we can just plot the bar charts for Kendall Tau.
        pass

    plt.close()


def plot_rank_correlation_bar(
    benchmark_results: List[ComparisonResult], 
    output_dir: str = "results"
):
    """
    Plot bar charts for Kendall Tau and Spearman Rho across strategies.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    names = [r.metric_name for r in benchmark_results]
    taus = [r.kendall_tau for r in benchmark_results]
    rhos = [r.spearman_rho for r in benchmark_results]
    top_10 = [r.top_10_accuracy for r in benchmark_results]
    
    df = pd.DataFrame({
        "Strategy": names * 3,
        "Metric Value": taus + rhos + top_10,
        "Metric Type": ["Kendall τ"] * len(names) + ["Spearman ρ"] * len(names) + ["Top-10 Acc"] * len(names)
    })

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Strategy", y="Metric Value", hue="Metric Type", palette="viridis")
    plt.title("Rank Correlation and Top-10 Accuracy of NAS Scoring Metrics")
    plt.ylabel("Score")
    plt.ylim(-0.2, 1.0)
    plt.legend(loc="upper left")
    
    out_file = os.path.join(output_dir, "rank_correlation.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved correlation bar plot to {out_file}")


def plot_search_trajectory_zoomed(csv_path: str, output_dir: str = "results", max_evals: int = 50):
    """
    Plot a zoomed-in version of the search trajectory to highlight early convergence.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    df_zoomed = df[df["evaluations"] <= max_evals]

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df_zoomed, 
        x="evaluations", 
        y="best_test_accuracy", 
        hue="strategy",
        errorbar=("ci", 95),
        linewidth=2.5,
        marker="o"
    )

    plt.title(f"Early Convergence (First {max_evals} Evaluations)")
    plt.xlabel("Number of Evaluated Architectures")
    plt.ylabel("Best Test Accuracy Found (%)")
    plt.legend(title="Fitness Strategy", loc="lower right")
    
    out_file = os.path.join(output_dir, "search_trajectory_zoomed.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved zoomed trajectory plot to {out_file}")


def plot_score_vs_width_binned(csv_path: str, output_dir: str = "results", dataset_name: str = "cifar10"):
    """
    Plot Tropical Scores vs Parameter Count using binned medians and IQR
    to clearly diagnose the bottleneck bound failure.
    Uses log-normalization for the scores to show exponential growth clearly.
    """
    import numpy as np
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # Log transform scores (add 1 to avoid log(0))
    df["log_generic"] = np.log10(df["tropical_generic"] + 1)
    df["log_tight"] = np.log10(df["tropical_tight"] + 1)
    
    # Bin by parameter count into 10 equal-sized buckets (quantiles)
    df["param_bin"] = pd.qcut(df["params"], q=10, duplicates="drop")
    # Get the mid-point of each bin for the x-axis plotting
    df["param_bin_mid"] = df["param_bin"].apply(lambda x: x.mid).astype(float)
    
    # Melt dataframe for seaborn
    df_melt = pd.melt(
        df, 
        id_vars=["param_bin_mid"], 
        value_vars=["log_generic", "log_tight"],
        var_name="Score Type", 
        value_name="Log10(Score)"
    )
    
    df_melt["Score Type"] = df_melt["Score Type"].map({
        "log_generic": "Tropical Generic (Montúfar)",
        "log_tight": "Tropical Tight (Serra)"
    })

    plt.figure(figsize=(10, 6))
    
    # Plot lineplot with median estimator and IQR error bars (pi, 50)
    sns.lineplot(
        data=df_melt, 
        x="param_bin_mid", 
        y="Log10(Score)", 
        hue="Score Type",
        estimator=np.median,
        errorbar=("pi", 50),
        marker="o",
        linewidth=2.5,
        err_style="band",
        palette=["#3498db", "#e74c3c"]
    )

    plt.title(f"Score vs Network Width (Params) - {dataset_name.upper()}\nDiagnosing Bottleneck Overcorrection (Binned Median ± IQR)")
    plt.xlabel("Parameter Count (Proxy for Width & Capacity)")
    plt.ylabel("Log10(Tropical Score)")
    
    out_file = os.path.join(output_dir, f"score_vs_width_binned_{dataset_name}.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved binned diagnostic plot to {out_file}")

