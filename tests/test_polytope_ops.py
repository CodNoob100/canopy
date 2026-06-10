"""
Tests for canopy.geometry.polytope_ops
=======================================

Verifies:
1. Minkowski sum correctness and vertex count
2. Volume computation against known results
3. Mixed volume via inclusion-exclusion
4. Vertex enumeration from halfspaces
5. PolytopeCache hit/miss behavior
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from canopy.geometry.polytope_ops import (
    PolytopeCache,
    minkowski_sum,
    mixed_volume,
    polytope_volume,
    vertex_enumeration_from_halfspaces,
)


# ===================================================================
#  Volume
# ===================================================================

class TestPolytopeVolume:
    def test_unit_square(self):
        """Unit square in 2D: area = 1."""
        verts = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]
        ])
        assert_allclose(polytope_volume(verts), 1.0, atol=1e-10)

    def test_unit_cube(self):
        """Unit cube in 3D: volume = 1."""
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ], dtype=float)
        assert_allclose(polytope_volume(verts), 1.0, atol=1e-10)

    def test_triangle(self):
        """Triangle with vertices (0,0), (1,0), (0,1): area = 0.5."""
        verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        assert_allclose(polytope_volume(verts), 0.5, atol=1e-10)

    def test_degenerate_collinear(self):
        """Collinear points in 2D: volume = 0."""
        verts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        assert polytope_volume(verts) == 0.0

    def test_single_point(self):
        """Single point: volume = 0."""
        assert polytope_volume(np.array([[1.0, 2.0]])) == 0.0


# ===================================================================
#  Minkowski sum
# ===================================================================

class TestMinkowskiSum:
    def test_two_segments_1d(self):
        """[0,1] + [0,2] = [0,3] in 1D."""
        A = np.array([[0.0], [1.0]])
        B = np.array([[0.0], [2.0]])
        result = minkowski_sum(A, B)
        assert result.shape[1] == 1
        assert_allclose(np.min(result), 0.0, atol=1e-10)
        assert_allclose(np.max(result), 3.0, atol=1e-10)

    def test_square_plus_square(self):
        """
        Unit square + unit square = [0,2]×[0,2] square.
        Volume should be 4.
        """
        sq = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        result = minkowski_sum(sq, sq)
        vol = polytope_volume(result)
        assert_allclose(vol, 4.0, atol=1e-10)

    def test_triangle_plus_point(self):
        """Triangle + {(1,1)} = translated triangle."""
        tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        pt = np.array([[1.0, 1.0]])
        result = minkowski_sum(tri, pt)
        assert len(result) == 3
        expected = tri + np.array([1.0, 1.0])
        # Check that result vertices match expected (up to ordering)
        for v in expected:
            dists = np.linalg.norm(result - v, axis=1)
            assert np.min(dists) < 1e-10

    def test_dimension_mismatch_raises(self):
        A = np.array([[0.0, 0.0]])
        B = np.array([[0.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="Dimension mismatch"):
            minkowski_sum(A, B)


# ===================================================================
#  Mixed volume
# ===================================================================

class TestMixedVolume:
    def test_two_segments_2d(self):
        """
        Mixed volume of two line segments in 2D:
        P1 = Conv{(0,0), (1,0)},  P2 = Conv{(0,0), (0,1)}.
        MV = |det([e1 | e2])| = 1.
        """
        P1 = np.array([[0.0, 0.0], [1.0, 0.0]])
        P2 = np.array([[0.0, 0.0], [0.0, 1.0]])
        mv = mixed_volume([P1, P2])
        assert_allclose(mv, 1.0, atol=1e-8)

    def test_same_polytope(self):
        """
        MV(P, P) = 2! × Vol(P) = 2 × Vol(P) for 2D.
        For the unit square: MV = 2 × 1 = 2.
        """
        sq = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        mv = mixed_volume([sq, sq])
        assert_allclose(mv, 2.0, atol=1e-8)

    def test_parallel_segments(self):
        """
        Two parallel segments have mixed volume 0.
        P1 = Conv{(0,0), (1,0)},  P2 = Conv{(0,0), (2,0)}.
        """
        P1 = np.array([[0.0, 0.0], [1.0, 0.0]])
        P2 = np.array([[0.0, 0.0], [2.0, 0.0]])
        mv = mixed_volume([P1, P2])
        assert_allclose(mv, 0.0, atol=1e-8)


# ===================================================================
#  Vertex enumeration
# ===================================================================

class TestVertexEnumeration:
    def test_unit_square(self):
        """
        Unit square: 0 ≤ x ≤ 1, 0 ≤ y ≤ 1.
        H-rep: x ≥ 0, -x ≥ -1, y ≥ 0, -y ≥ -1
        → A = [[-1,0],[1,0],[0,-1],[0,1]], b = [0,1,0,1]
        """
        A = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=float)
        b = np.array([0, 1, 0, 1], dtype=float)
        verts = vertex_enumeration_from_halfspaces(A, b)
        assert verts is not None
        assert len(verts) == 4
        vol = polytope_volume(verts)
        assert_allclose(vol, 1.0, atol=1e-8)

    def test_triangle(self):
        """
        Triangle: x ≥ 0, y ≥ 0, x + y ≤ 1.
        """
        A = np.array([[-1, 0], [0, -1], [1, 1]], dtype=float)
        b = np.array([0, 0, 1], dtype=float)
        verts = vertex_enumeration_from_halfspaces(A, b)
        assert verts is not None
        assert len(verts) == 3


# ===================================================================
#  PolytopeCache
# ===================================================================

class TestPolytopeCache:
    def test_cache_hit(self):
        cache = PolytopeCache(max_size=10)
        verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        vol1 = cache.volume_cached(verts)
        vol2 = cache.volume_cached(verts)
        assert vol1 == vol2
        assert cache.hits == 1
        assert cache.misses == 1

    def test_cache_eviction(self):
        cache = PolytopeCache(max_size=2)
        v1 = np.array([[0.0], [1.0]])
        v2 = np.array([[0.0], [2.0]])
        v3 = np.array([[0.0], [3.0]])
        cache.volume_cached(v1)
        cache.volume_cached(v2)
        cache.volume_cached(v3)  # should evict v1
        assert len(cache) == 2

    def test_order_independence(self):
        """Same vertices in different order should hit cache."""
        cache = PolytopeCache()
        v1 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        v2 = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        vol1 = cache.volume_cached(v1)
        vol2 = cache.volume_cached(v2)
        assert vol1 == vol2
        assert cache.hits == 1  # second call was a hit

    def test_hit_rate(self):
        cache = PolytopeCache()
        v = np.array([[0.0], [1.0]])
        cache.volume_cached(v)
        cache.volume_cached(v)
        cache.volume_cached(v)
        assert cache.hit_rate == pytest.approx(2 / 3)

    def test_clear(self):
        cache = PolytopeCache()
        v = np.array([[0.0], [1.0]])
        cache.volume_cached(v)
        cache.clear()
        assert len(cache) == 0
        assert cache.hits == 0

    def test_minkowski_sum_cached(self):
        cache = PolytopeCache()
        A = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        B = np.array([[0.0, 0.0], [1.0, 1.0]])
        r1 = cache.minkowski_sum_cached(A, B)
        r2 = cache.minkowski_sum_cached(A, B)
        assert np.array_equal(r1, r2)
        assert cache.hits == 1
