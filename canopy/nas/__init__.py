"""
canopy.nas — Neural Architecture Search Pipeline
==================================================

NAS system using tropical expressivity as the fitness function.

Modules:
    architecture    — ArchitectureSpec and search space definitions
    tropical_score  — Scoring function wrapping tropical geometry
    search          — Evolutionary and gradient-based search strategies
    benchmark       — NAS-Bench integration and baseline comparison
"""

from canopy.nas.architecture import (
    ArchitectureSpec,
    SearchSpace,
    NASBench201SearchSpace,
)
from canopy.nas.tropical_score import tropical_expressivity_score
from canopy.nas.search import EvolutionarySearch

__all__ = [
    "ArchitectureSpec",
    "SearchSpace",
    "NASBench201SearchSpace",
    "tropical_expressivity_score",
    "EvolutionarySearch",
]
