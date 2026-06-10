"""
Cell Counting Algorithms for ReLU Networks
===========================================

Three strategies for counting the number of linear regions of a
feedforward ReLU network, plus a unified dispatcher.

Strategies
----------
1. ActivationPatternCounter  — Exact via sign-vector LP enumeration.
   Practical for ≤ 20 total hidden neurons.
   Time:  O(2^M × LP(n, M)),  M = total hidden neurons.

2. LayerWiseCounter  — Layer-by-layer enumeration exploiting depth
   structure.  Exact for narrow layers (n_l ≤ 15 each).
   Time:  O(Σ_l |S_l| × 2^{n_l} × LP(n_0, C_l))
   where S_l = feasible states at layer l, C_l = accumulated constraints.

3. SamplingCounter  — Monte Carlo + Chao1 nonparametric estimator.
   Runs in O(N × M) regardless of network size.
   Produces a lower bound with optional confidence interval.

References
----------
- Hanin & Rolnick (2019) "Complexity of Linear Regions in Deep NNs"
- Serra, Tjandraatmadja, Ramalingam (2018) "Bounding and Counting
  Linear Regions of Deep Neural Networks"
- Chao (1984) "Nonparametric estimation of the number of classes
  in a population"
"""

from __future__ import annotations

import numpy as np
from collections import Counter
from dataclasses import dataclass, field
from itertools import product as iterproduct
from typing import Dict, List, Literal, Optional, Tuple


# ---------------------------------------------------------------------------
#  Activation pattern utilities
# ---------------------------------------------------------------------------

def get_activation_pattern(
    x: np.ndarray,
    layer_weights: List[np.ndarray],
    layer_biases: List[np.ndarray],
) -> Tuple[int, ...]:
    """
    Compute the activation pattern of a single input x.

    Returns a tuple of 0/1 values, one per hidden neuron, indicating
    whether each ReLU is active (1) or inactive (0).

    Time:  O(Σ_l n_l × n_{l-1}).
    Space: O(Σ_l n_l).
    """
    x = np.asarray(x, dtype=np.float64)
    pattern: List[int] = []
    a = x

    # Process all hidden layers (all except the last)
    for W, b in zip(layer_weights[:-1], layer_biases[:-1]):
        z = W @ a + b
        bits = (z > 0).astype(int)
        pattern.extend(bits.tolist())
        a = np.maximum(z, 0.0)

    return tuple(pattern)


def get_activation_patterns_batch(
    X: np.ndarray,
    layer_weights: List[np.ndarray],
    layer_biases: List[np.ndarray],
) -> np.ndarray:
    """
    Compute activation patterns for a batch of inputs.

    Parameters
    ----------
    X : ndarray, shape (B, n_0)
        Batch of inputs.
    layer_weights : list of ndarray
        [W¹, …, W^L].
    layer_biases : list of ndarray
        [b¹, …, b^L].

    Returns
    -------
    ndarray, shape (B, M), dtype uint8
        Binary activation patterns.  M = total hidden neurons.

    Time:  O(B × Σ_l n_l × n_{l-1}).
    Space: O(B × M).
    """
    X = np.asarray(X, dtype=np.float64)
    parts: List[np.ndarray] = []
    a = X  # (B, n_{l-1})

    for W, b in zip(layer_weights[:-1], layer_biases[:-1]):
        z = a @ W.T + b  # (B, n_l)
        active = (z > 0).astype(np.uint8)
        parts.append(active)
        a = z * active  # ReLU

    if len(parts) == 0:
        return np.zeros((X.shape[0], 0), dtype=np.uint8)
    return np.hstack(parts)


def _network_forward(
    x: np.ndarray,
    layer_weights: List[np.ndarray],
    layer_biases: List[np.ndarray],
) -> float:
    """Forward pass returning scalar output (for verification)."""
    a = np.asarray(x, dtype=np.float64)
    for l, (W, b) in enumerate(zip(layer_weights, layer_biases)):
        z = W @ a + b
        if l < len(layer_weights) - 1:
            a = np.maximum(z, 0.0)
        else:
            a = z
    return float(a[0]) if a.ndim > 0 else float(a)


# ---------------------------------------------------------------------------
#  ActivationPatternCounter — exact enumeration
# ---------------------------------------------------------------------------

class ActivationPatternCounter:
    """
    Exact linear region counting via sign-vector LP enumeration.

    For each possible activation pattern σ ∈ {0,1}^M (where M is the
    total number of hidden neurons), we check whether the polyhedron

        {x ∈ R^{n_0} : for all j, σ_j · (w_j · x + b_j) > 0}

    has non-empty interior.  This is an LP feasibility problem.

    Complexity
    ----------
    Time:  O(2^M × LP(n_0, M)).   LP via HiGHS is ~O(M × n_0).
    Space: O(M × n_0).
    Practical for  M ≤ 20.
    """

    def __init__(self, max_neurons: int = 20):
        """
        Parameters
        ----------
        max_neurons : int
            Refuse to run if total hidden neurons exceed this.
        """
        self.max_neurons = max_neurons

    def count(
        self,
        layer_weights: List[np.ndarray],
        layer_biases: List[np.ndarray],
        input_dim: int,
    ) -> int:
        """Count exact linear regions."""
        hidden_widths = [W.shape[0] for W in layer_weights[:-1]]
        total_hidden = sum(hidden_widths)

        if total_hidden > self.max_neurons:
            raise ValueError(
                f"Network has {total_hidden} hidden neurons > max "
                f"{self.max_neurons}. Use LayerWiseCounter or SamplingCounter."
            )
        if total_hidden == 0:
            return 1

        count = 0
        for bits in iterproduct([0, 1], repeat=total_hidden):
            if self._is_pattern_feasible(
                bits, hidden_widths, layer_weights, layer_biases, input_dim
            ):
                count += 1
        return count

    def _is_pattern_feasible(
        self,
        pattern: Tuple[int, ...],
        hidden_widths: List[int],
        layer_weights: List[np.ndarray],
        layer_biases: List[np.ndarray],
        input_dim: int,
    ) -> bool:
        """
        Check if an activation pattern is feasible.

        Given pattern σ = (σ_1^1, ..., σ_{n_1}^1, σ_1^2, ..., σ_{n_2}^2, ...),
        check if ∃ x ∈ R^{n_0} such that the network with input x
        produces exactly this activation pattern.

        We compose the affine maps layer by layer using the pattern,
        building constraints  A·x ≥ b  (or  A·x ≤ b  for inactive neurons).
        Then check LP feasibility.
        """
        from scipy.optimize import linprog

        n = input_dim
        # Variables: [x_1, ..., x_n, t]   (Chebyshev-style: maximize t)
        # Constraints:  sign_j · (a_j · x + c_j) ≥ t   for all j
        # Rewrite:  -sign_j · a_j · x + t ≤ sign_j · c_j

        # Build the affine maps layer by layer
        # A_eff, c_eff: z_l = A_eff · x + c_eff (pre-activation of layer l)
        constraint_A = []  # (n_0,) normals
        constraint_b = []  # scalars

        offset = 0
        A_eff = np.eye(n)  # maps input x to current "activation output"
        c_eff = np.zeros(n)

        for l, (W, b) in enumerate(zip(layer_weights[:-1], layer_biases[:-1])):
            n_l = W.shape[0]
            sigma_l = np.array(pattern[offset : offset + n_l], dtype=np.float64)
            offset += n_l

            # Pre-activation: z = W @ a + b = W @ (A_eff @ x + c_eff) + b
            #                    = (W @ A_eff) @ x + (W @ c_eff + b)
            Z_A = W @ A_eff   # (n_l, n_0)
            Z_c = W @ c_eff + b  # (n_l,)

            # Constraints for this layer
            for j in range(n_l):
                s = 2.0 * sigma_l[j] - 1.0  # +1 if active, -1 if inactive
                constraint_A.append(s * Z_A[j])
                constraint_b.append(-s * Z_c[j])
                # Constraint:  s · (Z_A[j] · x + Z_c[j]) ≥ 0
                # → s · Z_A[j] · x ≥ -s · Z_c[j]
                # In linprog form (A_ub @ x ≤ b_ub):
                #   -s · Z_A[j] · x ≤ s · Z_c[j]

            # Update effective map: activation output = diag(sigma) @ z
            D = np.diag(sigma_l)
            A_eff = D @ Z_A   # (n_l, n_0)
            c_eff = D @ Z_c   # (n_l,)

        if len(constraint_A) == 0:
            return True

        # Solve Chebyshev LP: maximize t s.t. s_j(a_j·x + c_j) ≥ t
        C_A = np.array(constraint_A)  # (M, n_0)
        C_b = np.array(constraint_b)  # (M,)
        M_cons = len(C_b)

        # LP: minimize -t  (maximize slack t)
        # Variables: [x (n), t (1)]
        # Constraints:  C_A[i] · x - t ≥ C_b[i]  →  -C_A[i] · x + t ≤ -C_b[i]
        # Bound t ∈ [0, 1] to prevent unboundedness.
        c_obj = np.zeros(n + 1)
        c_obj[-1] = -1.0  # maximize t

        A_ub = np.hstack([-C_A, np.ones((M_cons, 1))])
        b_ub = C_b

        bounds = [(None, None)] * n + [(0.0, 1.0)]  # t bounded

        result = linprog(
            c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs"
        )

        if result.success:
            t_opt = result.x[-1]
            return t_opt > 1e-10
        return False




# ---------------------------------------------------------------------------
#  LayerWiseCounter — depth-exploiting enumeration
# ---------------------------------------------------------------------------

@dataclass
class _RegionState:
    """
    Internal state for layer-wise enumeration.

    Tracks the accumulated constraints and the effective affine map
    from the input space to the current layer's activation output.
    """
    constraint_normals: np.ndarray   # (C, n_0)
    constraint_offsets: np.ndarray   # (C,)  — normals · x ≥ offsets
    effective_A: np.ndarray          # (n_l, n_0)
    effective_c: np.ndarray          # (n_l,)


class LayerWiseCounter:
    """
    Layer-by-layer linear region counting.

    Exploits the depth structure of feedforward networks by processing
    one layer at a time.  At each layer, enumerate all 2^{n_l} activation
    patterns and check feasibility only against accumulated constraints
    (not the full exponential space).

    This is exact and equivalent to ActivationPatternCounter, but much
    faster when individual layers are narrow — even if the total neuron
    count is large.

    Complexity
    ----------
    Let S_l = number of feasible states after processing layer l.
    Time:  O(Σ_l  S_l × 2^{n_l} × LP(n_0, C_l))
    Space: O(max_l S_l × C_l × n_0)

    Practical when  max_l n_l ≤ 15.  For very deep networks, S_l
    can still grow exponentially; use max_states to cap it.
    """

    def __init__(self, max_width: int = 15, max_states: int = 100_000):
        """
        Parameters
        ----------
        max_width : int
            Maximum layer width to enumerate (refuse wider layers).
        max_states : int
            Cap on the number of tracked states.  If exceeded, the
            counter returns the current count as a lower bound.
        """
        self.max_width = max_width
        self.max_states = max_states

    def count(
        self,
        layer_weights: List[np.ndarray],
        layer_biases: List[np.ndarray],
        input_dim: int,
    ) -> int:
        """
        Count linear regions layer by layer.

        Returns
        -------
        int
            Exact count (or lower bound if max_states was hit).
        """
        from scipy.optimize import linprog

        n = input_dim
        hidden_layers = list(zip(layer_weights[:-1], layer_biases[:-1]))

        if len(hidden_layers) == 0:
            return 1

        # Check layer widths
        for l, (W, b) in enumerate(hidden_layers):
            if W.shape[0] > self.max_width:
                raise ValueError(
                    f"Layer {l} has width {W.shape[0]} > max_width={self.max_width}."
                )

        # Initial state: no constraints, identity map
        states: List[_RegionState] = [
            _RegionState(
                constraint_normals=np.zeros((0, n)),
                constraint_offsets=np.zeros(0),
                effective_A=np.eye(n),
                effective_c=np.zeros(n),
            )
        ]

        for l, (W, b) in enumerate(hidden_layers):
            n_l = W.shape[0]
            new_states: List[_RegionState] = []

            for state in states:
                # Pre-activation: z = W @ a + b = W @ (A·x + c) + b
                Z_A = W @ state.effective_A      # (n_l, n_0)
                Z_c = W @ state.effective_c + b  # (n_l,)

                for bits in iterproduct([0, 1], repeat=n_l):
                    sigma = np.array(bits, dtype=np.float64)

                    # Build new constraints for this activation pattern
                    new_normals = np.empty((n_l, n))
                    new_offsets = np.empty(n_l)

                    for j in range(n_l):
                        s = 2.0 * sigma[j] - 1.0
                        new_normals[j] = s * Z_A[j]
                        new_offsets[j] = -s * Z_c[j]

                    # Combined constraints
                    all_normals = np.vstack([state.constraint_normals, new_normals]) \
                        if state.constraint_normals.shape[0] > 0 else new_normals
                    all_offsets = np.concatenate([state.constraint_offsets, new_offsets]) \
                        if state.constraint_offsets.shape[0] > 0 else new_offsets

                    # Feasibility check: ∃ x s.t. all_normals · x ≥ all_offsets?
                    # LP: maximize t s.t. all_normals[i] · x - t ≥ all_offsets[i]
                    C = all_normals.shape[0]

                    c_obj = np.zeros(n + 1)
                    c_obj[-1] = -1.0

                    A_ub = np.hstack([-all_normals, np.ones((C, 1))])
                    b_ub = all_offsets

                    bounds = [(None, None)] * n + [(0.0, 1.0)]
                    result = linprog(
                        c_obj, A_ub=A_ub, b_ub=b_ub,
                        bounds=bounds,
                        method="highs",
                    )

                    if result.success and result.x[-1] > 1e-10:
                        # Feasible — create new state
                        D = np.diag(sigma)
                        new_A = D @ Z_A
                        new_c = D @ Z_c

                        new_states.append(_RegionState(
                            constraint_normals=all_normals,
                            constraint_offsets=all_offsets,
                            effective_A=new_A,
                            effective_c=new_c,
                        ))

                if len(new_states) >= self.max_states:
                    # Safety valve: return current count as lower bound
                    return len(new_states)

            states = new_states

        return len(states)


# ---------------------------------------------------------------------------
#  SamplingCounter — Monte Carlo estimation
# ---------------------------------------------------------------------------

@dataclass
class SamplingResult:
    """Result of a sampling-based region count estimation."""
    observed: int          # Number of distinct patterns observed
    chao1_estimate: float  # Chao1 lower-bound estimate
    n_samples: int         # Total samples drawn
    singletons: int        # Patterns seen exactly once
    doubletons: int        # Patterns seen exactly twice
    coverage: float        # Good-Turing sample coverage estimate

    @property
    def is_saturated(self) -> bool:
        """True if sampling has likely found all regions (coverage ≈ 1)."""
        return self.coverage > 0.999

    def __repr__(self) -> str:
        return (
            f"SamplingResult(observed={self.observed}, "
            f"chao1={self.chao1_estimate:.0f}, "
            f"coverage={self.coverage:.4f}, "
            f"n_samples={self.n_samples})"
        )


class SamplingCounter:
    """
    Monte Carlo linear region estimation with Chao1 nonparametric bounds.

    Algorithm
    ---------
    1. Sample N inputs uniformly from a hypersphere of radius R.
    2. For each input, compute the activation pattern.
    3. Count distinct patterns (= observed regions).
    4. Use the Chao1 estimator for a lower bound on total regions.
    5. Compute Good-Turing sample coverage.

    The Chao1 estimator (Chao, 1984):

        Ŝ = S_obs + f₁² / (2 f₂)

    where f₁ = singletons, f₂ = doubletons.  This is a nonparametric
    lower bound on the true number of species (activation patterns).

    Complexity
    ----------
    Time:  O(N × Σ_l n_l × n_{l-1}).
    Space: O(N × M)  where M = total hidden neurons.
    Deterministic runtime regardless of architecture size.
    """

    def __init__(
        self,
        n_samples: int = 10_000,
        radius: float = 10.0,
        seed: Optional[int] = None,
    ):
        self.n_samples = n_samples
        self.radius = radius
        self.seed = seed

    def count(
        self,
        layer_weights: List[np.ndarray],
        layer_biases: List[np.ndarray],
        input_dim: int,
    ) -> int:
        """Return the Chao1 estimate (rounded to int)."""
        result = self.count_detailed(layer_weights, layer_biases, input_dim)
        return int(np.ceil(result.chao1_estimate))

    def count_detailed(
        self,
        layer_weights: List[np.ndarray],
        layer_biases: List[np.ndarray],
        input_dim: int,
    ) -> SamplingResult:
        """
        Full sampling analysis with statistics.
        """
        rng = np.random.default_rng(self.seed)

        # Sample from hypersphere of given radius
        # Generate Gaussian vectors and normalize, then scale
        X = rng.standard_normal((self.n_samples, input_dim))
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        # Uniform on sphere × uniform radial scaling
        radii = self.radius * rng.uniform(0, 1, (self.n_samples, 1)) ** (
            1.0 / input_dim
        )
        X = X / norms * radii

        # Compute activation patterns
        patterns = get_activation_patterns_batch(X, layer_weights, layer_biases)

        # Count frequencies
        pattern_tuples = [tuple(row) for row in patterns]
        freq = Counter(pattern_tuples)

        S_obs = len(freq)
        counts = list(freq.values())
        freq_of_freq = Counter(counts)

        f1 = freq_of_freq.get(1, 0)  # singletons
        f2 = freq_of_freq.get(2, 0)  # doubletons

        # Chao1 estimator
        if f2 > 0:
            chao1 = S_obs + f1 ** 2 / (2 * f2)
        elif f1 > 0:
            chao1 = S_obs + f1 * (f1 - 1) / 2
        else:
            chao1 = float(S_obs)

        # Good-Turing coverage: 1 - f1/N
        coverage = 1.0 - f1 / self.n_samples

        return SamplingResult(
            observed=S_obs,
            chao1_estimate=chao1,
            n_samples=self.n_samples,
            singletons=f1,
            doubletons=f2,
            coverage=coverage,
        )


# ---------------------------------------------------------------------------
#  Unified API
# ---------------------------------------------------------------------------

def count_linear_regions(
    layer_weights: List[np.ndarray],
    layer_biases: List[np.ndarray],
    input_dim: int,
    *,
    method: Literal["auto", "exact", "layerwise", "sampling"] = "auto",
    n_samples: int = 10_000,
    seed: Optional[int] = None,
) -> int:
    """
    Count the number of linear regions of a feedforward ReLU network.

    Selects the best algorithm based on network size:
    - 'exact':     ActivationPatternCounter  (M ≤ 20)
    - 'layerwise': LayerWiseCounter          (max n_l ≤ 15)
    - 'sampling':  SamplingCounter           (any size)
    - 'auto':      Pick the best feasible method.

    Parameters
    ----------
    layer_weights, layer_biases : list of ndarray
        Network parameters.  layer_weights[l] has shape (n_{l+1}, n_l).
    input_dim : int
        Input dimension.
    method : str
        Algorithm selection.
    n_samples : int
        Sample count for 'sampling' method.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    int
        Number of linear regions (exact or estimate depending on method).
    """
    hidden_widths = [W.shape[0] for W in layer_weights[:-1]]
    total_hidden = sum(hidden_widths)
    max_width = max(hidden_widths) if hidden_widths else 0

    if method == "auto":
        if total_hidden == 0:
            return 1
        elif total_hidden <= 20:
            method = "exact"
        elif max_width <= 15:
            method = "layerwise"
        else:
            method = "sampling"

    if method == "exact":
        counter = ActivationPatternCounter(max_neurons=25)
        return counter.count(layer_weights, layer_biases, input_dim)

    elif method == "layerwise":
        counter = LayerWiseCounter(max_width=max(max_width, 16))
        return counter.count(layer_weights, layer_biases, input_dim)

    elif method == "sampling":
        counter = SamplingCounter(n_samples=n_samples, seed=seed)
        return counter.count(layer_weights, layer_biases, input_dim)

    else:
        raise ValueError(f"Unknown method: {method!r}")
