"""
CANOPY — Tropical Geometry-Guided Neural Architecture Search
============================================================

A NAS system where the fitness function is derived from Tropical Geometry:
the exact count of linear regions of a ReLU network architecture,
computable analytically, before training.

Subpackages:
    canopy.math      — Tropical algebra and Newton polytope theory
    canopy.geometry   — Computational geometry kernels (Phase 2)
    canopy.nas        — NAS pipeline and benchmarking (Phase 3)
"""

__version__ = "0.1.0"
