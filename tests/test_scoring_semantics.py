"""Guard the interpretation and reproducibility of expressivity scores."""

import numpy as np

from canopy.math.tropical_theory import (
    generic_linear_region_count,
    score_conv_expressivity,
)
from canopy.nas.architecture import ArchitectureSpec
from canopy.nas.tropical_score import tropical_expressivity_score


def test_generic_score_returns_integer_bound_not_log10():
    arch = ArchitectureSpec(layer_widths=(2, 4, 1))
    score = tropical_expressivity_score(arch, mode="generic")
    assert score == 11
    assert isinstance(score, int)


def test_deep_generic_bound_matches_zaslavsky_product():
    assert generic_linear_region_count([1, 2, 2, 1]) == 9


def test_conv_proxy_does_not_mutate_global_numpy_rng():
    before = np.random.get_state()
    try:
        first = score_conv_expressivity(n=3, k=2, C=1)
        after = np.random.get_state()
        assert first == score_conv_expressivity(n=3, k=2, C=1)
        for lhs, rhs in zip(before, after):
            np.testing.assert_array_equal(lhs, rhs)
    finally:
        np.random.set_state(before)
