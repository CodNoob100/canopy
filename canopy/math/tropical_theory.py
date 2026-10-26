"""
Tropical Geometry Theory for ReLU Networks
==========================================

Mathematical Foundation
-----------------------
Every ReLU network computes a tropical rational function. This module
formalizes the map from network weights to tropical polynomial representation
and provides an architecture-level expressivity formula.

Tropical Semiring (max-plus algebra):
    (R ∪ {-∞},  ⊕,  ⊙)
    a ⊕ b  =  max(a, b)         (tropical addition)
    a ⊙ b  =  a + b             (tropical multiplication)

A tropical polynomial in x ∈ R^n:
    p(x) = ⊕_i (c_i ⊙ x^{α_i})
         = max_i (c_i + α_i · x)

where c_i ∈ R (coefficients) and α_i ∈ R^n (exponent vectors).

Key identities used throughout:
    1. ReLU(z) = max(0, z)                                   — tropical poly
    2. max_i(f_i) + max_j(g_j) = max_{i,j}(f_i + g_j)       — conventional sum = tropical product
    3. max(0, p(x) - q(x)) = (p ⊕ q)(x) - q(x)             — ReLU of tropical rational
    4. c · max_i(f_i) = max_i(c · f_i)  for c > 0            — positive scaling

References
----------
- Zhang et al. (2018) "Tropical Geometry of Deep Neural Networks"
- Maragos et al. (2021) "Tropical Geometry and Machine Learning"
- Montúfar et al. (2014) "On the Number of Linear Regions of Deep Neural Networks"
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from math import comb
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
#  TropicalPolynomial
# ---------------------------------------------------------------------------

@dataclass
class TropicalPolynomial:
    """
    A tropical polynomial  p(x) = max_i (c_i + α_i · x).

    This is a convex piecewise-linear function — equivalently the upper
    envelope of a finite family of affine functions {x ↦ c_i + α_i · x}.

    Attributes
    ----------
    coefficients : ndarray, shape (T,)
        Constant terms c_i of each affine piece.
    exponents : ndarray, shape (T, n)
        Linear-coefficient vectors α_i.  Each row is one exponent vector.

    Invariant
    ---------
    coefficients.shape[0] == exponents.shape[0]  (= T, the term count)
    exponents.shape[1] == n  (input dimension, consistent across operations)
    """

    coefficients: np.ndarray
    exponents: np.ndarray

    # -- construction helpers ------------------------------------------------

    def __post_init__(self):
        self.coefficients = np.asarray(self.coefficients, dtype=np.float64)
        self.exponents = np.asarray(self.exponents, dtype=np.float64)
        if self.coefficients.ndim != 1:
            raise ValueError(
                f"coefficients must be 1-D, got shape {self.coefficients.shape}"
            )
        if self.exponents.ndim != 2:
            raise ValueError(
                f"exponents must be 2-D, got shape {self.exponents.shape}"
            )
        if self.coefficients.shape[0] != self.exponents.shape[0]:
            raise ValueError(
                f"Term count mismatch: {self.coefficients.shape[0]} coefficients "
                f"vs {self.exponents.shape[0]} exponent vectors"
            )

    @staticmethod
    def zero(input_dim: int) -> TropicalPolynomial:
        """The constant-zero function: p(x) = 0  ∀x."""
        return TropicalPolynomial(
            coefficients=np.array([0.0]),
            exponents=np.zeros((1, input_dim)),
        )

    @staticmethod
    def constant(value: float, input_dim: int) -> TropicalPolynomial:
        """A constant function: p(x) = value  ∀x."""
        return TropicalPolynomial(
            coefficients=np.array([value]),
            exponents=np.zeros((1, input_dim)),
        )

    @staticmethod
    def identity(index: int, input_dim: int) -> TropicalPolynomial:
        """The coordinate projection x_i as a single-term tropical polynomial."""
        exp = np.zeros((1, input_dim))
        exp[0, index] = 1.0
        return TropicalPolynomial(coefficients=np.array([0.0]), exponents=exp)

    # -- properties ----------------------------------------------------------

    @property
    def num_terms(self) -> int:
        """Number of affine pieces (monomials)."""
        return self.coefficients.shape[0]

    @property
    def input_dim(self) -> int:
        """Dimension of the input space."""
        return self.exponents.shape[1]

    # -- evaluation ----------------------------------------------------------

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate p(x) = max_i (c_i + α_i · x).

        Parameters
        ----------
        x : array-like, shape (n,)

        Returns
        -------
        float
        """
        x = np.asarray(x, dtype=np.float64)
        return float(np.max(self.coefficients + self.exponents @ x))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorised evaluation over a batch of points.

        Parameters
        ----------
        X : ndarray, shape (B, n)

        Returns
        -------
        ndarray, shape (B,)
        """
        X = np.asarray(X, dtype=np.float64)
        # (T, n) @ (n, B) → (T, B), then max over axis 0
        return np.max(self.coefficients[:, None] + self.exponents @ X.T, axis=0)

    def active_term(self, x: np.ndarray) -> int:
        """Return the index of the term achieving the maximum at x."""
        x = np.asarray(x, dtype=np.float64)
        return int(np.argmax(self.coefficients + self.exponents @ x))

    # -- tropical operations -------------------------------------------------

    def tropical_sum(self, other: TropicalPolynomial) -> TropicalPolynomial:
        """
        Tropical addition: (p ⊕ q)(x) = max(p(x), q(x)).

        Implementation: union of terms.
        Complexity: O(T_p + T_q) time and space.
        """
        if self.input_dim != other.input_dim:
            raise ValueError(
                f"Dimension mismatch: {self.input_dim} vs {other.input_dim}"
            )
        return TropicalPolynomial(
            coefficients=np.concatenate([self.coefficients, other.coefficients]),
            exponents=np.vstack([self.exponents, other.exponents]),
        )

    def conventional_sum(self, other: TropicalPolynomial) -> TropicalPolynomial:
        """
        Conventional (real-valued) addition:  (p + q)(x) = p(x) + q(x).

        Equals the tropical product  p ⊙ q  because:
            max_i(a_i + α_i·x) + max_j(b_j + β_j·x)
            = max_{i,j} ((a_i + b_j) + (α_i + β_j)·x)

        Implementation: Cartesian product of terms.
        Complexity: O(T_p · T_q · n) time and space.

        WARNING: term count multiplies — use judiciously on large polynomials.
        """
        if self.input_dim != other.input_dim:
            raise ValueError(
                f"Dimension mismatch: {self.input_dim} vs {other.input_dim}"
            )
        n = self.input_dim
        # Outer sum of coefficients  (T_p, 1) + (1, T_q) → (T_p, T_q)
        new_coeffs = (
            self.coefficients[:, None] + other.coefficients[None, :]
        ).ravel()
        # Outer sum of exponents  (T_p, 1, n) + (1, T_q, n) → (T_p, T_q, n)
        new_exps = (
            self.exponents[:, None, :] + other.exponents[None, :, :]
        ).reshape(-1, n)
        return TropicalPolynomial(coefficients=new_coeffs, exponents=new_exps)

    def scale(self, c: float) -> TropicalPolynomial:
        """
        Multiply by a *positive* scalar:
            (c · p)(x) = c · max_i(a_i + α_i·x) = max_i(c·a_i + c·α_i·x).

        For c = 0 returns the constant-zero polynomial.
        Raises ValueError for c < 0 (flips convexity — use TropicalRational).

        Complexity: O(T · n).
        """
        if c < 0:
            raise ValueError(
                "scale() requires c ≥ 0.  Negative scaling destroys the max "
                "structure; handle it at the TropicalRational level."
            )
        if c == 0.0:
            return TropicalPolynomial.zero(self.input_dim)
        return TropicalPolynomial(
            coefficients=c * self.coefficients,
            exponents=c * self.exponents,
        )

    def shift(self, c: float) -> TropicalPolynomial:
        """
        Add a constant:  (p + c)(x) = max_i((c_i + c) + α_i · x).

        Complexity: O(T).
        """
        return TropicalPolynomial(
            coefficients=self.coefficients + c,
            exponents=self.exponents.copy(),
        )

    # -- pruning (dominated-term removal) ------------------------------------

    def prune_dominated(self, tol: float = 1e-12) -> TropicalPolynomial:
        """
        Remove dominated terms that never achieve the maximum.

        A term (c_i, α_i) is **dominated** if ∀ x:
            c_i + α_i · x  <  max_{j ≠ i} (c_j + α_j · x)

        Equivalently, term i is active iff the polyhedron
            { x : (α_i - α_j)·x ≥ c_j - c_i  ∀ j ≠ i }
        is non-empty.  We check feasibility via scipy.optimize.linprog.

        Complexity: O(T² · LP(n, T))  where LP is the linear-program cost.
        """
        if self.num_terms <= 1:
            return TropicalPolynomial(
                self.coefficients.copy(), self.exponents.copy()
            )

        from scipy.optimize import linprog

        T, n = self.num_terms, self.input_dim
        active = np.ones(T, dtype=bool)

        for i in range(T):
            if not active[i]:
                continue
            others = np.where(active & (np.arange(T) != i))[0]
            if len(others) == 0:
                continue

            # Feasibility:  ∃ x  s.t.  (α_i − α_j) · x ≥ c_j − c_i  ∀ j ∈ others
            # Rewrite as    A_ub @ x ≤ b_ub   (scipy convention)
            A_ub = -(self.exponents[i] - self.exponents[others])  # (|others|, n)
            b_ub = -(self.coefficients[others] - self.coefficients[i]) + tol

            result = linprog(
                c=np.zeros(n),
                A_ub=A_ub,
                b_ub=b_ub,
                bounds=(None, None),
                method="highs",
            )
            if not result.success:
                active[i] = False

        return TropicalPolynomial(
            coefficients=self.coefficients[active].copy(),
            exponents=self.exponents[active].copy(),
        )

    def deduplicate(self, tol: float = 1e-12) -> TropicalPolynomial:
        """
        Merge duplicate exponent vectors, keeping the largest coefficient.

        If two terms share (approximately) the same exponent vector,
        only the one with the higher coefficient survives — because
        max(a + α·x, b + α·x) = max(a, b) + α·x.

        Complexity: O(T² · n)  (pairwise comparison; could be improved with hashing).
        """
        if self.num_terms <= 1:
            return TropicalPolynomial(
                self.coefficients.copy(), self.exponents.copy()
            )

        T, n = self.num_terms, self.input_dim
        used = np.zeros(T, dtype=bool)
        new_coeffs = []
        new_exps = []

        for i in range(T):
            if used[i]:
                continue
            # Find all terms with the same exponent as i
            diffs = np.abs(self.exponents[i] - self.exponents[i + 1 :]).max(axis=1)
            duplicates = np.where(diffs < tol)[0] + (i + 1)
            group = np.concatenate([[i], duplicates])
            used[group] = True

            best = group[np.argmax(self.coefficients[group])]
            new_coeffs.append(self.coefficients[best])
            new_exps.append(self.exponents[best])

        return TropicalPolynomial(
            coefficients=np.array(new_coeffs),
            exponents=np.array(new_exps).reshape(-1, n),
        )

    def __repr__(self) -> str:
        return (
            f"TropicalPolynomial(terms={self.num_terms}, "
            f"input_dim={self.input_dim})"
        )


# ---------------------------------------------------------------------------
#  TropicalRational
# ---------------------------------------------------------------------------

@dataclass
class TropicalRational:
    """
    A tropical rational function  f(x) = p(x) − q(x).

    Here p (numerator) and q (denominator) are tropical polynomials.
    This is the canonical form for any continuous piecewise-linear (CPWL)
    function — and therefore for any ReLU-network output.

    Linear regions of f correspond to cells in the common refinement
    of the upper-envelope decompositions of p and q.  The number of
    such cells is at most  T_p × T_q.

    Attributes
    ----------
    numerator : TropicalPolynomial   (p)
    denominator : TropicalPolynomial (q)
    """

    numerator: TropicalPolynomial
    denominator: TropicalPolynomial

    def __post_init__(self):
        if self.numerator.input_dim != self.denominator.input_dim:
            raise ValueError(
                f"Dimension mismatch: numerator has dim {self.numerator.input_dim}, "
                f"denominator has dim {self.denominator.input_dim}"
            )

    # -- properties ----------------------------------------------------------

    @property
    def input_dim(self) -> int:
        return self.numerator.input_dim

    @property
    def max_linear_regions(self) -> int:
        """
        Upper bound on the number of linear regions:  T_p × T_q.

        This is tight when the normal fans of p's and q's Newton polytopes
        are in general position.
        """
        return self.numerator.num_terms * self.denominator.num_terms

    # -- evaluation ----------------------------------------------------------

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate f(x) = p(x) − q(x)."""
        return self.numerator.evaluate(x) - self.denominator.evaluate(x)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Vectorised evaluation over a batch of points."""
        return self.numerator.evaluate_batch(X) - self.denominator.evaluate_batch(X)

    # -- core operations -----------------------------------------------------

    def relu(self) -> TropicalRational:
        """
        Apply ReLU:  max(0, f(x)) = max(0, p(x) − q(x)).

        Theorem
        -------
        max(0, p − q) = (p ⊕ q) − q.

        Proof:
            Case p(x) ≥ q(x):  (p ⊕ q)(x) = p(x)  ⟹  result = p − q  ✓
            Case p(x) < q(x):  (p ⊕ q)(x) = q(x)  ⟹  result = 0      ✓

        Complexity: O(T_p + T_q)  time and space.
        """
        return TropicalRational(
            numerator=self.numerator.tropical_sum(self.denominator),
            denominator=TropicalPolynomial(
                self.denominator.coefficients.copy(),
                self.denominator.exponents.copy(),
            ),
        )

    def prune(self) -> TropicalRational:
        """Prune dominated terms in both components."""
        return TropicalRational(
            numerator=self.numerator.prune_dominated(),
            denominator=self.denominator.prune_dominated(),
        )

    def deduplicate(self) -> TropicalRational:
        """Merge duplicate exponent vectors in both components."""
        return TropicalRational(
            numerator=self.numerator.deduplicate(),
            denominator=self.denominator.deduplicate(),
        )

    # -- constructors --------------------------------------------------------

    @staticmethod
    def from_variable(index: int, input_dim: int) -> TropicalRational:
        """
        Input variable x_i as a tropical rational.
            x_i = p(x) − q(x)   with  p(x) = x_i, q(x) = 0.
        """
        return TropicalRational(
            numerator=TropicalPolynomial.identity(index, input_dim),
            denominator=TropicalPolynomial.zero(input_dim),
        )

    @staticmethod
    def from_constant(value: float, input_dim: int) -> TropicalRational:
        """
        A scalar constant as a tropical rational.
            value ≥ 0  ⟹  p = value, q = 0.
            value < 0  ⟹  p = 0, q = |value|.
        """
        if value >= 0:
            return TropicalRational(
                numerator=TropicalPolynomial.constant(value, input_dim),
                denominator=TropicalPolynomial.zero(input_dim),
            )
        return TropicalRational(
            numerator=TropicalPolynomial.zero(input_dim),
            denominator=TropicalPolynomial.constant(-value, input_dim),
        )

    def __repr__(self) -> str:
        return (
            f"TropicalRational(num_terms={self.numerator.num_terms}, "
            f"den_terms={self.denominator.num_terms}, "
            f"input_dim={self.input_dim})"
        )


# ---------------------------------------------------------------------------
#  Layer propagation
# ---------------------------------------------------------------------------

def _reduce_conventional_sum(
    polys: List[TropicalPolynomial], input_dim: int
) -> TropicalPolynomial:
    """
    Compute  p_1 + p_2 + ... + p_k  (conventional / real-valued sum).

    Each "+" is a tropical product (Cartesian product of terms), so the
    total term count is  T_1 × T_2 × ... × T_k.

    Returns the constant-zero polynomial when the list is empty (additive
    identity: 0 + f = f).
    """
    if len(polys) == 0:
        return TropicalPolynomial.zero(input_dim)
    result = polys[0]
    for p in polys[1:]:
        result = result.conventional_sum(p)
    return result


def propagate_linear(
    inputs: List[TropicalRational],
    weights: np.ndarray,
    biases: np.ndarray,
) -> List[TropicalRational]:
    """
    Propagate through a **linear** layer (no activation).

        z_j = Σ_k  w_{jk} · a_k  +  b_j

    where each  a_k = p_k(x) − q_k(x)  is a tropical rational function.

    Derivation
    ----------
    Split weights into positive (K⁺) and negative (K⁻) subsets.

        z_j  =  [Σ_{k∈K⁺} w_{jk} p_k  +  Σ_{k∈K⁻} |w_{jk}| q_k]      ← P_j
              − [Σ_{k∈K⁺} w_{jk} q_k  +  Σ_{k∈K⁻} |w_{jk}| p_k]      ← Q_j
              + b_j

    Each conventional Σ is a tropical ⊙ (Cartesian product of terms).
    Bias b_j is folded in via coefficient shift on the appropriate component.

    Parameters
    ----------
    inputs : list of TropicalRational, length K
        Tropical rational representations of the K input activations.
    weights : ndarray, shape (m, K)
        Weight matrix.  Row j, column k gives w_{jk}.
    biases : ndarray, shape (m,)
        Bias vector.

    Returns
    -------
    list of TropicalRational, length m
        One tropical rational per output neuron (pre-activation).

    Complexity
    ----------
    Time per output neuron: O(Π_k T_k)  where T_k is the term count of
    the k-th input's numerator/denominator (whichever is larger).
    """
    weights = np.asarray(weights, dtype=np.float64)
    biases = np.asarray(biases, dtype=np.float64)
    m, K = weights.shape
    n = inputs[0].input_dim
    outputs: List[TropicalRational] = []

    for j in range(m):
        pos_num: List[TropicalPolynomial] = []  # terms going into P_j
        neg_num: List[TropicalPolynomial] = []  # terms going into Q_j

        for k in range(K):
            w = weights[j, k]
            if abs(w) < 1e-15:
                continue

            p_k = inputs[k].numerator
            q_k = inputs[k].denominator

            if w > 0:
                pos_num.append(p_k.scale(w))
                neg_num.append(q_k.scale(w))
            else:
                pos_num.append(q_k.scale(-w))
                neg_num.append(p_k.scale(-w))

        P_j = _reduce_conventional_sum(pos_num, n)
        Q_j = _reduce_conventional_sum(neg_num, n)

        # Fold in bias
        b = biases[j]
        if b >= 0:
            P_j = P_j.shift(b)
        else:
            Q_j = Q_j.shift(-b)

        outputs.append(TropicalRational(numerator=P_j, denominator=Q_j))

    return outputs


def propagate_relu_layer(
    inputs: List[TropicalRational],
    weights: np.ndarray,
    biases: np.ndarray,
) -> List[TropicalRational]:
    """
    Propagate through a linear layer followed by ReLU activation.

        a_j = ReLU(Σ_k w_{jk} a_k + b_j)

    This is ``propagate_linear`` followed by ``.relu()`` on each output.
    """
    pre = propagate_linear(inputs, weights, biases)
    return [z.relu() for z in pre]


# ---------------------------------------------------------------------------
#  Full architecture → tropical rational
# ---------------------------------------------------------------------------

def architecture_to_tropical(
    layer_weights: List[np.ndarray],
    layer_biases: List[np.ndarray],
    input_dim: int,
    *,
    apply_relu_to_last: bool = False,
    prune_per_layer: bool = False,
) -> List[TropicalRational]:
    """
    Map a feedforward ReLU network (given explicit weights) to its
    tropical rational representation.

    Parameters
    ----------
    layer_weights : list of ndarray
        [W¹, W², …, W^L]  where W^l has shape (n_l, n_{l-1}).
    layer_biases : list of ndarray
        [b¹, b², …, b^L]  where b^l has shape (n_l,).
    input_dim : int
        Dimension n₀ of the input space.
    apply_relu_to_last : bool
        Whether to apply ReLU after the final layer.
    prune_per_layer : bool
        If True, prune dominated terms after every layer.  This keeps
        term counts manageable at the cost of O(T² · LP) per layer.

    Returns
    -------
    list of TropicalRational
        One tropical rational function per output neuron of the last layer.

    Complexity
    ----------
    Without pruning, term counts grow as  O(2^{n_1} · 2^{n_2} · … )
    — doubly exponential in depth.  With pruning, the count is bounded
    by the number of linear regions of the sub-network up to that layer,
    which is at most  Π_l Σ_{j≤n₀} C(n_l, j).
    """
    if len(layer_weights) != len(layer_biases):
        raise ValueError("Must have same number of weight and bias arrays.")

    # Input variables
    current: List[TropicalRational] = [
        TropicalRational.from_variable(i, input_dim) for i in range(input_dim)
    ]

    L = len(layer_weights)
    for l in range(L):
        W, b = layer_weights[l], layer_biases[l]
        is_last = l == L - 1

        if is_last and not apply_relu_to_last:
            current = propagate_linear(current, W, b)
        else:
            current = propagate_relu_layer(current, W, b)

        if prune_per_layer and not (is_last and not apply_relu_to_last):
            current = [a.prune() for a in current]

    return current


# ---------------------------------------------------------------------------
#  Architecture-level expressivity (no weights required)
# ---------------------------------------------------------------------------

def _zaslavsky_bound(num_hyperplanes: int, ambient_dim: int) -> int:
    """
    Zaslavsky's theorem: n hyperplanes in general position in R^d
    partition space into  Σ_{k=0}^{d} C(n, k)  regions.
    """
    return sum(comb(num_hyperplanes, k) for k in range(ambient_dim + 1))


def generic_linear_region_count(
    layer_widths: List[int],
    *,
    connectivity: Optional[np.ndarray] = None,
) -> int:
    """
    Compute a combinatorial *upper bound* on linear regions of a ReLU network.

    This architecture-only quantity is not the exact region count of any
    particular network; it may not be attained even with generic weights.

    Theorem (Montúfar et al. 2014; Serra et al. 2018)
    --------------------------------------------------
    For a network  x ∈ R^{n_0} → R^{n_L}  with hidden layers of widths
    n_1, …, n_{L-1}, an architecture-level upper bound is:

        R = Π_{l=1}^{L-1}  Σ_{j=0}^{min(n_l, n_0)}  C(n_l, j)

    where n_0 is the **input dimension**, not the preceding layer width.

    This uses the fact that the piecewise-linear map from R^{n_0} can
    have at most  Σ_{j=0}^{n_0} C(n_l, j)  activation patterns at
    layer l (by Zaslavsky's theorem applied to the n_l hyperplanes
    induced by the layer's neurons in the n_0-dimensional input space).

    The product over layers follows from the compositional structure:
    each layer can multiply the region count by its Zaslavsky bound.

    For bottleneck layers (n_l < n_0):
        The bound simplifies to  2^{n_l}  since
        Σ_{j=0}^{n_l} C(n_l, j) = 2^{n_l}.

    Parameters
    ----------
    layer_widths : list of int
        [n_0, n_1, …, n_L]  where n_0 is input dim, n_L is output dim,
        and n_1 … n_{L-1} are hidden-layer widths.
    connectivity : ndarray, optional
        Adjacency matrix for skip connections.  Not yet implemented;
        reserved for Phase 3.

    Returns
    -------
    int
        An integer architecture-level upper bound on linear regions.

    Examples
    --------
    >>> generic_linear_region_count([1, 4, 1])    # 1D input, one hidden layer of 4
    5
    >>> generic_linear_region_count([2, 3, 1])    # 2D input, one hidden layer of 3
    7
    >>> generic_linear_region_count([1, 2, 2, 1]) # 1D input, two hidden layers of 2
    9
    """
    if connectivity is not None:
        raise NotImplementedError(
            "Skip-connection aware region counting is planned for Phase 3."
        )

    if len(layer_widths) < 2:
        raise ValueError("Need at least input + output layer (≥ 2 entries).")

    n_0 = layer_widths[0]
    # Hidden layers are layer_widths[1:-1]
    hidden = layer_widths[1:-1]

    if len(hidden) == 0:
        # Pure linear network: 1 linear region
        return 1

    regions = 1
    for n_l in hidden:
        # Zaslavsky bound for n_l hyperplanes in R^{n_0}
        regions *= _zaslavsky_bound(n_l, n_0)

    return regions


def linear_region_upper_bound_tight(
    layer_widths: List[int],
) -> int:
    """
    Tighter upper bound using the **min of input dim and cumulative rank**.

    Theorem (Serra, Tjandraatmadja, Ramalingam 2018)
    -------------------------------------------------
    The effective ambient dimension for layer l is

        d_l = min(n_0, n_1, …, n_{l-1})

    because any bottleneck layer restricts the rank of the Jacobian.
    The tighter bound replaces n_0 with d_l:

        R ≤ Π_{l=1}^{L-1}  Σ_{j=0}^{min(n_l, d_l)}  C(n_l, j)

    Parameters
    ----------
    layer_widths : list of int
        [n_0, n_1, …, n_L].

    Returns
    -------
    int
    """
    if len(layer_widths) < 2:
        raise ValueError("Need at least input + output layer.")

    hidden = layer_widths[1:-1]
    if len(hidden) == 0:
        return 1

    regions = 1
    d = layer_widths[0]  # running effective dimension

    for l, n_l in enumerate(hidden):
        regions *= _zaslavsky_bound(n_l, d)
        # Update effective dimension: bottleneck
        d = min(d, n_l)

    return regions


# ---------------------------------------------------------------------------
#  Phase 7: Real Architectures (Convolutions and Skip Connections)
# ---------------------------------------------------------------------------

def score_conv_expressivity(n: int, k: int, C: int) -> int:
    """
    Computes a deterministic sampled arrangement proxy for a 1D Conv layer.
    Uses one Toeplitz weight sample; it does not maximize over all filters
    and must not be interpreted as an exact architecture-level maximum.
    
    Parameters
    ----------
    n : int
        Input spatial dimension.
    k : int
        Filter size (e.g., 3 for 3x3).
    C : int
        Number of output channels.
        
    Returns
    -------
    int
        Number of cells in the sampled hyperplane arrangement. This is a
        deterministic proxy, not a proven maximum over convolution filters.
    """
    import numpy as np
    from itertools import combinations
    
    m = n - k + 1
    if m <= 0:
        return 1
        
    # Generic weights
    rng = np.random.default_rng(42)
    W = np.zeros((C * m, n))
    for c in range(C):
        filter_w = rng.standard_normal(k)
        for i in range(m):
            W[c * m + i, i:i+k] = filter_w
            
    regions = 0
    # zaslavsky over independent subsets
    for j in range(n + 1):
        for sub_indices in combinations(range(C * m), j):
            if np.linalg.matrix_rank(W[list(sub_indices)]) == j:
                regions += 1
    return regions


def score_residual_expressivity(incoming_vertices_per_dim: List[int]) -> int:
    """
    Computes the upper bound of linear regions for a Residual Skip Connection block:
        y = ReLU(F(x) + x)
        
    Derived using the Minkowski sum of the Newton polygons of F(x) and the identity shift,
    followed by the convex hull with the origin (ReLU).
    
    Parameters
    ----------
    incoming_vertices_per_dim : list of int
        A list of length n, where each element is V_i, the upper bound of regions 
        for the F(x) branch in that dimension.
        
    Returns
    -------
    int
        The total linear region upper bound for the residual block.
    """
    import numpy as np
    return int(np.prod([v + 1 for v in incoming_vertices_per_dim]))
