"""
Tropical Expressivity Scoring Function
========================================

The black-box scoring function for NAS.  Wraps the tropical geometry
machinery from canopy.math and the cell-counting algorithms from
canopy.geometry into a single callable with the signature:

    score = tropical_expressivity_score(arch)

This is what the NAS search loop calls to evaluate candidate architectures.

Two modes:
    1. Weight-free (default):  Uses the Montúfar/Serra formula.
       O(L) time, no weights needed, computes the *generic* region count.
    2. Weight-aware:  Instantiates random weights and counts exact/approximate
       linear regions using the Phase 2 cell-counting kernel.
"""

from __future__ import annotations

import time
import numpy as np
from typing import List, Literal, Optional

from canopy.nas.architecture import ArchitectureSpec
from canopy.math.tropical_theory import (
    generic_linear_region_count,
    linear_region_upper_bound_tight,
)
from canopy.geometry.cell_counter import count_linear_regions


def tropical_expressivity_score(
    arch: ArchitectureSpec,
    *,
    mode: Literal["generic", "tight", "weight_aware"] = "generic",
    weights: Optional[List[np.ndarray]] = None,
    biases: Optional[List[np.ndarray]] = None,
    n_random_inits: int = 1,
    counting_method: str = "auto",
    seed: Optional[int] = None,
) -> int:
    """
    Compute the tropical expressivity score for an architecture.

    This is the fitness function for NAS.  Returns the (exact or estimated)
    number of linear regions the architecture can produce.

    Parameters
    ----------
    arch : ArchitectureSpec
        The architecture to score.
    mode : str
        Scoring mode:
        - 'generic':  Montúfar formula.  Fastest.  O(L).
        - 'tight':    Serra et al. tight bound with bottleneck awareness.
        - 'weight_aware':  Instantiate weights and count regions.
    weights, biases : list of ndarray, optional
        Explicit weights for 'weight_aware' mode.  If None, random
        weights are generated.
    n_random_inits : int
        For 'weight_aware' mode without explicit weights: number of
        random weight initializations.  Returns the median count.
    counting_method : str
        Passed to count_linear_regions() for 'weight_aware' mode.
    seed : int, optional
        Random seed for weight generation.

    Returns
    -------
    int
        The expressivity score (linear region count or bound).
    """
    if mode == "generic":
        return generic_linear_region_count(list(arch.layer_widths))

    elif mode == "tight":
        return linear_region_upper_bound_tight(list(arch.layer_widths))

    elif mode == "weight_aware":
        return _score_weight_aware(
            arch,
            weights=weights,
            biases=biases,
            n_random_inits=n_random_inits,
            counting_method=counting_method,
            seed=seed,
        )

    else:
        raise ValueError(f"Unknown mode: {mode!r}")


def _score_weight_aware(
    arch: ArchitectureSpec,
    *,
    weights: Optional[List[np.ndarray]],
    biases: Optional[List[np.ndarray]],
    n_random_inits: int,
    counting_method: str,
    seed: Optional[int],
) -> int:
    """Weight-aware scoring with optional random initialization."""
    if weights is not None and biases is not None:
        return count_linear_regions(
            weights, biases, arch.input_dim, method=counting_method,
        )

    rng = np.random.default_rng(seed)
    counts = []

    for i in range(n_random_inits):
        w, b = _generate_random_weights(arch, rng)
        c = count_linear_regions(
            w, b, arch.input_dim, method=counting_method,
        )
        counts.append(c)

    return int(np.median(counts))


def _generate_random_weights(
    arch: ArchitectureSpec,
    rng: np.random.Generator,
) -> tuple:
    """
    Generate Kaiming-uniform random weights for an architecture.

    Uses fan_in scaling:  W ~ U(-1/√n_in, 1/√n_in).
    """
    weights = []
    biases = []
    for i in range(len(arch.layer_widths) - 1):
        n_in = arch.layer_widths[i]
        n_out = arch.layer_widths[i + 1]
        scale = 1.0 / np.sqrt(n_in) if n_in > 0 else 1.0
        W = rng.uniform(-scale, scale, size=(n_out, n_in))
        b = rng.uniform(-scale, scale, size=n_out)
        weights.append(W)
        biases.append(b)
    return weights, biases


def score_batch(
    architectures: List[ArchitectureSpec],
    *,
    mode: Literal["generic", "tight"] = "generic",
) -> List[int]:
    """
    Score a batch of architectures.  Only supports weight-free modes
    for efficiency.

    Parameters
    ----------
    architectures : list of ArchitectureSpec
    mode : str

    Returns
    -------
    list of int
        Scores in the same order.
    """
    return [
        tropical_expressivity_score(arch, mode=mode) for arch in architectures
    ]


def benchmark_scoring_speed(
    arch: ArchitectureSpec,
    *,
    mode: str = "generic",
    n_trials: int = 100,
) -> dict:
    """
    Benchmark the scoring function speed.

    Returns
    -------
    dict with keys: mean_ms, std_ms, max_ms, meets_target (< 100ms).
    """
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        tropical_expressivity_score(arch, mode=mode)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times_arr = np.array(times)
    return {
        "mean_ms": float(times_arr.mean()),
        "std_ms": float(times_arr.std()),
        "max_ms": float(times_arr.max()),
        "meets_target": float(times_arr.max()) < 100.0,
    }


def tropical_expressivity_score_darts(
    arch: ArchitectureSpec, 
    input_spatial_dim: int = 32, 
    base_channels: int = 16
) -> float:
    """
    Computes the Phase 7 extended Tropical Expressivity score for a DARTS architecture.
    
    This function evaluates the linear region bounds derived by the Theoretical Geometer
    for Convolutional and Residual (Skip Connection) layers.
    
    Because exact calculation of Toeplitz matroid independence is astronomically slow
    for full-scale networks (C=16, m=32 -> 512 hyperplanes), we use the derived 
    generic proxy: a Conv layer behaves tropically as a dense layer with width C * m,
    and skip connections multiply the incoming regions via the Minkowski sum formula.
    
    Parameters
    ----------
    arch : ArchitectureSpec
        The DARTS architecture (must contain cell_ops like 'sep_conv_3x3', 'skip_connect').
    input_spatial_dim : int
        The sequence/spatial length.
    base_channels : int
        The base channel count.
        
    Returns
    -------
    float
        The log10 of the estimated tropical expressivity score.
    """
    if arch.cell_ops is None:
        raise ValueError("DARTS scoring requires cell_ops.")
        
    from canopy.math.tropical_theory import _zaslavsky_bound, score_residual_expressivity
    
    # 1. Base operations expressivity lookup
    # DARTS edges map to filter sizes / operations
    op_to_filter = {
        "none": 0,
        "skip_connect": -1,  # Special flag
        "max_pool_3x3": 0,   # Pooling doesn't add linear boundaries like relu/conv
        "avg_pool_3x3": 0,
        "sep_conv_3x3": 3,
        "sep_conv_5x5": 5,
        "dil_conv_3x3": 3,
        "dil_conv_5x5": 5,
    }
    
    # Simple DAG resolution for DARTS 4-node cell (14 edges, but let's assume a generic sequence for proxy)
    # The DARTS cell has 4 intermediate nodes. 
    # For a fast tropical proxy score, we trace the longest path of expressivity.
    
    total_log_regions = 0.0
    current_dim = input_spatial_dim
    current_v = [1] * current_dim  # Vertices per dimension tracker
    
    for op_entry in arch.cell_ops:
        if "~" in op_entry:
            op = op_entry.split("~")[0]
        else:
            op = op_entry
            
        k = op_to_filter.get(op, 0)
        
        if k == -1:
            # Skip connection!
            # Use the mathematically looser Cartesian product bound for y = ReLU(F(x) + x)
            # R_res = \prod (V_i + 1)
            # In log space: \sum log10(V_i + 1)
            # We explicitly use this looser bound instead of the tighter face-lattice bound
            # because empirical validation shows it acts as a superior structural proxy (ρ=0.51).
            total_log_regions += np.sum(np.log10([v + 1 for v in current_v]))
            # Update vertices for next layers (proxy increase)
            current_v = [v + 1 for v in current_v]
            
        elif k > 0:
            # Convolutional Layer!
            m = current_dim - k + 1
            if m <= 0: m = 1
            
            # The geometer's proxy: Conv layer behaves as dense layer of width C*m
            # Regions = ZaslavskyBound(num_hyperplanes=C*m, ambient_dim=current_dim)
            num_hyperplanes = base_channels * m
            
            # Since Zaslavsky can be large, we use an asymptotic log approximation:
            # log10(C(M, n)) ~ n log10(M/n * e) for large M
            if num_hyperplanes > current_dim:
                # Approximation of log10(comb(C*m, n))
                log_regions = current_dim * np.log10((num_hyperplanes / current_dim) * np.e)
                total_log_regions += log_regions
                
            # Update spatial dim
            current_dim = m
            
    return total_log_regions

from math import comb

def score_attention_regions(N_seq: int, d_model: int, H_heads: int) -> float:
    """
    Computes proxy upper bound of linear regions for a Multi-Head Attention layer.
    Derived using the zero-temperature Hard Attention (argmax) limit mapped to a 
    polyhedral piecewise-linear proxy via Zaslavsky's theorem on hyperplane arrangements.
    
    Returns
    -------
    float
        The log10 of the total regions.
    """
    if H_heads <= 0 or d_model <= 0:
        return 0.0
        
    d_k = d_model // H_heads
    
    # Number of boundary hyperplanes for Hard Attention routing per query
    hyperplanes_per_head = N_seq * (N_seq - 1) // 2
    
    if hyperplanes_per_head <= 0:
        return 0.0
    
    # Regions per head for a single query using Zaslavsky's bound
    # Using log-space approximation to prevent numeric overflow
    if hyperplanes_per_head > d_k and d_k > 0:
        log_regions_per_query_per_head = d_k * np.log10((hyperplanes_per_head / d_k) * np.e)
    else:
        limit = min(d_k, hyperplanes_per_head)
        regions = sum(comb(hyperplanes_per_head, i) for i in range(limit + 1))
        log_regions_per_query_per_head = np.log10(regions) if regions > 0 else 0.0
        
    # Total regions per query across all heads (intersection -> multiplication of regions -> addition in log space)
    log_regions_per_query = log_regions_per_query_per_head * H_heads
    
    # Total regions for the entire sequence (independent queries)
    total_log_regions = log_regions_per_query * N_seq
    
    return float(total_log_regions)
