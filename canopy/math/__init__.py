"""
canopy.math — Tropical Algebra for ReLU Networks
=================================================

Core mathematical machinery mapping ReLU network architectures
to tropical polynomials and Newton polytopes.
"""

from canopy.math.tropical_theory import (
    TropicalPolynomial,
    TropicalRational,
    propagate_linear,
    propagate_relu_layer,
    architecture_to_tropical,
    generic_linear_region_count,
)
from canopy.math.newton_polytope import NewtonPolytope

__all__ = [
    "TropicalPolynomial",
    "TropicalRational",
    "propagate_linear",
    "propagate_relu_layer",
    "architecture_to_tropical",
    "generic_linear_region_count",
    "NewtonPolytope",
]
