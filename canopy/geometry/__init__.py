"""
canopy.geometry — Computational Geometry Kernels
=================================================

Efficient cell counting and polytope operations for
tropical polynomial analysis of ReLU networks.
"""

from canopy.geometry.cell_counter import (
    ActivationPatternCounter,
    LayerWiseCounter,
    SamplingCounter,
    count_linear_regions,
    get_activation_patterns_batch,
)
from canopy.geometry.polytope_ops import (
    PolytopeCache,
    minkowski_sum,
    mixed_volume,
    polytope_volume,
)

__all__ = [
    "ActivationPatternCounter",
    "LayerWiseCounter",
    "SamplingCounter",
    "count_linear_regions",
    "get_activation_patterns_batch",
    "PolytopeCache",
    "minkowski_sum",
    "mixed_volume",
    "polytope_volume",
]
