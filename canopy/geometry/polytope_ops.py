"""
Efficient Polytope Operations with Caching
===========================================

Wrapper around scipy.spatial (primary) with optional pycddlib backend
for vertex enumeration.  All operations include explicit complexity
annotations and are designed for reuse across NAS architecture evaluations.

Key operations
--------------
- Minkowski sum via vertex enumeration
- Mixed volume via inclusion-exclusion (BKK bound)
- Polytope volume via ConvexHull
- H-rep → V-rep vertex enumeration
- LRU cache keyed by polytope vertex hash
"""

from __future__ import annotations

import hashlib
import numpy as np
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
#  Volume computation
# ---------------------------------------------------------------------------

def polytope_volume(vertices: np.ndarray) -> float:
    """
    Compute the volume of the convex hull of a set of points.

    Parameters
    ----------
    vertices : ndarray, shape (V, n)
        Point set whose convex hull volume is computed.

    Returns
    -------
    float
        Volume of Conv(vertices).  Returns 0 for degenerate
        configurations (fewer points than dimension + 1, or
        points in a lower-dimensional affine subspace).

    Complexity
    ----------
    Time:  O(V^{⌊n/2⌋})  via Quickhull (scipy.spatial.ConvexHull).
    Space: O(V · n).
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    V, n = vertices.shape

    if V <= n:
        return 0.0

    # Check rank
    centered = vertices - vertices[0]
    rank = np.linalg.matrix_rank(centered, tol=1e-10)
    if rank < n:
        return 0.0

    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(vertices)
        return float(hull.volume)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
#  Minkowski sum
# ---------------------------------------------------------------------------

def minkowski_sum(
    verts_a: np.ndarray,
    verts_b: np.ndarray,
    *,
    take_hull: bool = True,
) -> np.ndarray:
    """
    Compute the Minkowski sum  P + Q = {a + b : a ∈ P, b ∈ Q}.

    Uses the vertex-enumeration approach: the vertices of P + Q are a
    subset of {v_a + v_b : v_a ∈ vert(P), v_b ∈ vert(Q)}.  We compute
    the full Cartesian sum and then take the convex hull to obtain the
    actual vertices.

    Parameters
    ----------
    verts_a : ndarray, shape (V_a, n)
        Vertices of polytope P.
    verts_b : ndarray, shape (V_b, n)
        Vertices of polytope Q.  Must have same ambient dimension n.
    take_hull : bool
        If True (default), return only the convex hull vertices.
        If False, return the full Cartesian sum (useful when the caller
        needs all candidate points).

    Returns
    -------
    ndarray, shape (V_out, n)
        Vertices of the Minkowski sum P + Q.

    Complexity
    ----------
    Time:  O(V_a · V_b · n)  for the Cartesian sum, plus
           O((V_a · V_b)^{⌊n/2⌋})  for the convex hull.
    Space: O(V_a · V_b · n).
    """
    verts_a = np.asarray(verts_a, dtype=np.float64)
    verts_b = np.asarray(verts_b, dtype=np.float64)

    if verts_a.shape[1] != verts_b.shape[1]:
        raise ValueError(
            f"Dimension mismatch: {verts_a.shape[1]} vs {verts_b.shape[1]}"
        )

    # Cartesian sum: (V_a, 1, n) + (1, V_b, n) → (V_a, V_b, n)
    n = verts_a.shape[1]
    all_sums = (
        verts_a[:, None, :] + verts_b[None, :, :]
    ).reshape(-1, n)

    if not take_hull or all_sums.shape[0] <= n + 1:
        return all_sums

    # Deduplicate
    all_sums = np.unique(np.round(all_sums, decimals=12), axis=0)

    if all_sums.shape[0] <= n:
        return all_sums

    try:
        from scipy.spatial import ConvexHull
        centered = all_sums - all_sums[0]
        rank = np.linalg.matrix_rank(centered, tol=1e-10)
        if rank < 2:
            # Collinear or degenerate
            return all_sums
        if rank < n:
            # Project to subspace, compute hull, map back
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            projected = centered @ Vt[:rank].T
            hull = ConvexHull(projected)
            return all_sums[hull.vertices]
        hull = ConvexHull(all_sums)
        return all_sums[hull.vertices]
    except Exception:
        return all_sums


# ---------------------------------------------------------------------------
#  Mixed volume (BKK bound)
# ---------------------------------------------------------------------------

def mixed_volume(polytopes: List[np.ndarray]) -> float:
    """
    Compute the mixed volume of n polytopes in R^n via inclusion-exclusion.

    The mixed volume MV(P_1, …, P_n) is the unique symmetric multilinear
    function satisfying  MV(P, P, …, P) = n! · Vol(P).

    By inclusion-exclusion (Minkowski's theorem):

        MV(P_1,…,P_n) = Σ_{S ⊆ [n], S≠∅} (-1)^{n-|S|} · Vol(Σ_{i∈S} P_i)

    where the sum Σ_{i∈S} P_i is the Minkowski sum.

    The mixed volume provides the **BKK bound** on the number of
    solutions to a generic system of polynomial equations — and thus
    bounds linear region counts in the tropical framework.

    Parameters
    ----------
    polytopes : list of ndarray
        Each element has shape (V_i, n).  Must have exactly n polytopes.

    Returns
    -------
    float
        The mixed volume.  Non-negative by Alexandrov-Fenchel.

    Complexity
    ----------
    Time:  O(2^n × Minkowski_cost × ConvexHull_cost).
    Space: O(max_V^n × n)  for the iterated Minkowski sums.
    Practical for n ≤ 8.
    """
    n = len(polytopes)
    if n == 0:
        return 0.0

    # Check ambient dimensions match
    dim = polytopes[0].shape[1]
    if n != dim:
        raise ValueError(
            f"Need exactly n={dim} polytopes for ambient dimension {dim}, "
            f"got {n}."
        )

    total = 0.0
    indices = list(range(n))

    for size in range(1, n + 1):
        for subset in combinations(indices, size):
            # Compute Minkowski sum of polytopes in this subset
            mink = polytopes[subset[0]].copy()
            for idx in subset[1:]:
                mink = minkowski_sum(mink, polytopes[idx])

            vol = polytope_volume(mink)
            sign = (-1) ** (n - size)
            total += sign * vol

    return max(total, 0.0)  # Clamp numerical noise


# ---------------------------------------------------------------------------
#  Vertex enumeration (H-rep → V-rep)
# ---------------------------------------------------------------------------

def vertex_enumeration_from_halfspaces(
    A: np.ndarray,
    b: np.ndarray,
    interior_point: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Convert halfspace representation {x : A·x ≤ b} to vertex representation.

    Uses scipy.spatial.HalfspaceIntersection, which requires an interior
    point.  If not provided, one is found via LP.

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Normal vectors of halfspaces (pointing inward: A·x ≤ b).
    b : ndarray, shape (m,)
        Halfspace bounds.
    interior_point : ndarray, shape (n,), optional
        A strictly feasible point.  Found via LP if not given.

    Returns
    -------
    ndarray, shape (V, n), or None
        Vertices of the polytope, or None if infeasible / unbounded.

    Complexity
    ----------
    Time:  O(m^{⌊n/2⌋})  via dual convex hull.
    Space: O(m · n).
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m, n = A.shape

    if interior_point is None:
        # Find interior point via Chebyshev center LP
        from scipy.optimize import linprog

        # Maximize t s.t. A·x + ||A_i|| · t ≤ b_i
        norms = np.linalg.norm(A, axis=1, keepdims=True)
        c_lp = np.zeros(n + 1)
        c_lp[-1] = -1.0  # maximize t

        A_lp = np.hstack([A, norms])
        result = linprog(c_lp, A_ub=A_lp, b_ub=b, bounds=(None, None), method="highs")

        if not result.success or result.x[-1] <= 1e-10:
            return None  # Infeasible or degenerate
        interior_point = result.x[:n]

    try:
        from scipy.spatial import HalfspaceIntersection, ConvexHull

        halfspaces = np.hstack([A, -b[:, None]])
        hs = HalfspaceIntersection(halfspaces, interior_point)
        hull = ConvexHull(hs.intersections)
        return hs.intersections[hull.vertices]
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  PolytopeCache
# ---------------------------------------------------------------------------

def _vertex_hash(vertices: np.ndarray, precision: int = 8) -> str:
    """Hash a vertex set (order-independent) for caching."""
    rounded = np.round(vertices, decimals=precision)
    # Sort rows lexicographically for order independence
    sorted_verts = rounded[np.lexsort(rounded.T[::-1])]
    return hashlib.sha256(sorted_verts.tobytes()).hexdigest()


class PolytopeCache:
    """
    LRU cache for polytope computations.

    Caches results of expensive operations (ConvexHull, Minkowski sum,
    volume) keyed by a hash of the vertex set.  Designed for the NAS
    search loop where architectures share sub-structures.

    Parameters
    ----------
    max_size : int
        Maximum number of cached entries.

    Complexity
    ----------
    Lookup: O(V · n · log V)  for hashing (sort + hash).
    """

    def __init__(self, max_size: int = 1024):
        self.max_size = max_size
        self._cache: OrderedDict[str, object] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[object]:
        """Retrieve a cached result, or None."""
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, key: str, value: object) -> None:
        """Store a result in the cache."""
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = value

    def volume_cached(self, vertices: np.ndarray) -> float:
        """Cached volume computation."""
        key = "vol:" + _vertex_hash(vertices)
        result = self.get(key)
        if result is not None:
            return result
        vol = polytope_volume(vertices)
        self.put(key, vol)
        return vol

    def minkowski_sum_cached(
        self, verts_a: np.ndarray, verts_b: np.ndarray
    ) -> np.ndarray:
        """Cached Minkowski sum."""
        key = "mink:" + _vertex_hash(verts_a) + ":" + _vertex_hash(verts_b)
        result = self.get(key)
        if result is not None:
            return result
        mink = minkowski_sum(verts_a, verts_b)
        self.put(key, mink)
        return mink

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        return (
            f"PolytopeCache(size={len(self)}/{self.max_size}, "
            f"hit_rate={self.hit_rate:.2%})"
        )
