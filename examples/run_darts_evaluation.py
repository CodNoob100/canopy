import os
import sys
import logging
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from canopy.nas.tropical_score import tropical_expressivity_score_darts
from canopy.nas.architecture import ArchitectureSpec

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("DARTS_EVAL")

# NAS-Bench-301 mapping to extract ops
# The configspace usually returns something like: 
# {'edge_normal_0': 'sep_conv_3x3', ...}
def sample_darts_ops(rng):
    ops = (
        "max_pool_3x3", "avg_pool_3x3", "skip_connect", 
        "sep_conv_3x3", "sep_conv_5x5", "dil_conv_3x3", 
        "dil_conv_5x5", "none"
    )
    return tuple(ops[rng.integers(len(ops))] for _ in range(14))

def run_evaluation(num_samples=5000):
    try:
        import nasbench301 as nb
        logger.info(f"Loading NAS-Bench-301 surrogate models...")
        models_1_0_dir = os.path.expanduser("~/nb301_models")
        performance_model = nb.load_ensemble(models_1_0_dir) 
    except Exception as e:
        logger.warning(f"NAS-Bench-301 API or models not available ({e}).")
        logger.warning("Falling back to simulated DARTS structural evaluation...")
        performance_model = None

    rng = np.random.default_rng(42)
    
    tropical_scores = []
    param_counts = [] # Mocked param counts for DARTS
    accuracies = []
    
    print(f"Sampling {num_samples} DARTS architectures...")
    
    OPS = ['max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5']
    from nasbench301.representations import Genotype

    for i in range(num_samples):
        # 1. Sample DARTS architecture properly (2 edges per node)
        normal = []
        for n_idx in range(4):
            choices = rng.choice(OPS, size=2, replace=True)
            nodes = rng.choice(range(n_idx + 2), size=2, replace=False)
            normal.append((choices[0], int(nodes[0])))
            normal.append((choices[1], int(nodes[1])))
            
        genotype = Genotype(normal=normal, normal_concat=[2,3,4,5], reduce=normal, reduce_concat=[2,3,4,5])
        
        # Map to 14-edge array for tropical score
        ops_14 = ['none'] * 14
        start = 0
        n = 2
        for n_idx in range(4):
            op1, idx1 = normal[2 * n_idx]
            op2, idx2 = normal[2 * n_idx + 1]
            ops_14[start + idx1] = op1
            ops_14[start + idx2] = op2
            start += n
            n += 1

        arch = ArchitectureSpec(layer_widths=[32]*5, cell_ops=ops_14)
        
        # 2. Tropical Expressivity Score
        t_score = tropical_expressivity_score_darts(arch)
        tropical_scores.append(t_score)
        
        # Proxy parameter count (for partial correlation control)
        # We assign 0 to none, 0 to skip, 1 to pool, 5 to conv3, 9 to conv5
        p_count = 0
        for op in ops_14:
            if "conv_3x3" in op: p_count += 5
            elif "conv_5x5" in op: p_count += 9
            elif "pool" in op: p_count += 1
        param_counts.append(p_count)
        
        # 3. Accuracy
        if performance_model is not None:
            # Query the true XGBoost surrogate model
            config_dict = performance_model.convert_representation(genotype, "genotype")
            
            # NAS-Bench-301 requires additional hyperparameters to be specified
            try:
                from nasbench301.api import fixed_hyperparameters
                full_config = {**fixed_hyperparameters, **config_dict}
                
                # CRITICAL: The NAS-Bench-301 1.0 XGBoost models were trained with init_channels=50 and layers=20. 
                # The fixed_hyperparameters incorrectly specifies 32 and 8.
                full_config['NetworkSelectorDatasetInfo:darts:init_channels'] = 50
                full_config['NetworkSelectorDatasetInfo:darts:layers'] = 20
                
                acc = performance_model.model.query(full_config)
            except Exception as e:
                logger.error(f"Failed to query model: {e}")
                acc = 10.0
        else:
            acc = 10.0
        
        accuracies.append(acc)
        
        if (i+1) % 1000 == 0:
            logger.info(f"  Processed {i+1}/{num_samples}")

    tropical_scores = np.array(tropical_scores)
    param_counts = np.array(param_counts)
    accuracies = np.array(accuracies)

    # Calculate Rank Correlations
    rho_trop, _ = stats.spearmanr(tropical_scores, accuracies)
    tau_trop, _ = stats.kendalltau(tropical_scores, accuracies)
    
    rho_param, _ = stats.spearmanr(param_counts, accuracies)
    tau_param, _ = stats.kendalltau(param_counts, accuracies)

    logger.info("\n--- Rank Correlation (Spearman ρ) ---")
    logger.info(f"Tropical Expressivity: {rho_trop:.3f}")
    logger.info(f"Parameter Count:       {rho_param:.3f}")

    logger.info("\n--- Rank Correlation (Kendall τ) ---")
    logger.info(f"Tropical Expressivity: {tau_trop:.3f}")
    logger.info(f"Parameter Count:       {tau_param:.3f}")

    # Plotting
    os.makedirs("results", exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.scatter(tropical_scores, accuracies, alpha=0.3, color='blue', s=10)
    ax1.set_title(f"Tropical Score vs NAS-Bench-301 Accuracy\nSpearman: {rho_trop:.2f}")
    ax1.set_xlabel("Log Tropical Expressivity (Proxy)")
    ax1.set_ylabel("Validation Accuracy (%)")
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(param_counts, accuracies, alpha=0.3, color='orange', s=10)
    ax2.set_title(f"Parameter Count vs NAS-Bench-301 Accuracy\nSpearman: {rho_param:.2f}")
    ax2.set_xlabel("Parameter Count (Proxy)")
    ax2.set_ylabel("Validation Accuracy (%)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = "results/darts_correlation.png"
    plt.savefig(plot_path, dpi=150)
    logger.info(f"\nSaved correlation plot to {plot_path}")


if __name__ == "__main__":
    run_evaluation(num_samples=5000)
