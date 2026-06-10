"""
Tests for canopy.nas.architecture
==================================

Verifies:
1. ArchitectureSpec creation, properties, and immutability
2. Vector encoding/decoding roundtrip
3. FeedforwardSearchSpace: random, mutate, crossover
4. NASBench201SearchSpace: enumeration, index roundtrip, ops-to-arch
"""

import numpy as np
import pytest

from canopy.nas.architecture import (
    ArchitectureSpec,
    FeedforwardSearchSpace,
    NASBench201SearchSpace,
)


# ===================================================================
#  ArchitectureSpec
# ===================================================================

class TestArchitectureSpec:
    def test_basic_creation(self):
        arch = ArchitectureSpec(layer_widths=(2, 8, 4, 1))
        assert arch.input_dim == 2
        assert arch.output_dim == 1
        assert arch.depth == 3
        assert arch.hidden_widths == (8, 4)
        assert arch.total_hidden_neurons == 12

    def test_frozen(self):
        arch = ArchitectureSpec(layer_widths=(2, 4, 1))
        with pytest.raises(AttributeError):
            arch.layer_widths = (2, 8, 1)

    def test_hashable(self):
        a = ArchitectureSpec(layer_widths=(2, 4, 1))
        b = ArchitectureSpec(layer_widths=(2, 4, 1))
        assert hash(a) == hash(b)
        assert a == b
        s = {a, b}
        assert len(s) == 1

    def test_total_params(self):
        arch = ArchitectureSpec(layer_widths=(3, 5, 2))
        # 3→5: 5*3+5=20, 5→2: 2*5+2=12 → total 32
        assert arch.total_params == 32

    def test_non_relu_raises(self):
        with pytest.raises(ValueError, match="relu"):
            ArchitectureSpec(layer_widths=(2, 4, 1), activation="sigmoid")

    def test_vector_roundtrip_simple(self):
        arch = ArchitectureSpec(layer_widths=(2, 8, 4, 1))
        vec = arch.to_vector()
        arch2 = ArchitectureSpec.from_vector(vec)
        assert arch == arch2

    def test_repr(self):
        arch = ArchitectureSpec(layer_widths=(2, 4, 1))
        r = repr(arch)
        assert "2" in r and "4" in r and "1" in r


# ===================================================================
#  FeedforwardSearchSpace
# ===================================================================

class TestFeedforwardSearchSpace:
    def setup_method(self):
        self.space = FeedforwardSearchSpace(
            input_dim=2, output_dim=1,
            min_depth=1, max_depth=4,
            min_width=2, max_width=16,
        )
        self.rng = np.random.default_rng(42)

    def test_random(self):
        arch = self.space.random(self.rng)
        assert arch.input_dim == 2
        assert arch.output_dim == 1
        assert 1 <= len(arch.hidden_widths) <= 4
        for w in arch.hidden_widths:
            assert 2 <= w <= 16

    def test_random_diversity(self):
        archs = {self.space.random(self.rng) for _ in range(50)}
        assert len(archs) > 5  # should generate diverse architectures

    def test_mutate_preserves_io(self):
        arch = self.space.random(self.rng)
        mutated = self.space.mutate(arch, self.rng)
        assert mutated.input_dim == 2
        assert mutated.output_dim == 1

    def test_mutate_changes_something(self):
        arch = ArchitectureSpec(layer_widths=(2, 8, 8, 1))
        changed = False
        for seed in range(20):
            rng = np.random.default_rng(seed)
            m = self.space.mutate(arch, rng)
            if m != arch:
                changed = True
                break
        assert changed, "Mutation should change the architecture"

    def test_crossover_preserves_io(self):
        a = self.space.random(self.rng)
        b = self.space.random(self.rng)
        child = self.space.crossover(a, b, self.rng)
        assert child.input_dim == 2
        assert child.output_dim == 1


# ===================================================================
#  NASBench201SearchSpace
# ===================================================================

class TestNASBench201SearchSpace:
    def setup_method(self):
        self.space = NASBench201SearchSpace(channels=16, num_classes=10)
        self.rng = np.random.default_rng(42)

    def test_total_architectures(self):
        assert self.space.TOTAL_ARCHITECTURES == 15625

    def test_random_has_cell_ops(self):
        arch = self.space.random(self.rng)
        assert arch.cell_ops is not None
        assert len(arch.cell_ops) == 6
        for op in arch.cell_ops:
            assert op in self.space.OPS

    def test_index_roundtrip(self):
        for idx in [0, 1, 100, 15624]:
            arch = self.space.index_to_arch(idx)
            assert self.space.arch_to_index(arch) == idx

    def test_mutate_changes_one_edge(self):
        arch = self.space.random(self.rng)
        mutated = self.space.mutate(arch, self.rng)
        assert mutated.cell_ops is not None
        # At most one edge changed
        diffs = sum(
            1 for a, b in zip(arch.cell_ops, mutated.cell_ops) if a != b
        )
        assert diffs <= 1

    def test_crossover(self):
        a = self.space.random(self.rng)
        b = self.space.random(self.rng)
        child = self.space.crossover(a, b, self.rng)
        assert child.cell_ops is not None
        # Each edge comes from one parent
        for i in range(6):
            assert child.cell_ops[i] in (a.cell_ops[i], b.cell_ops[i])

    def test_enumerate_subset(self):
        """Enumerate first 10 architectures — should all be valid."""
        for idx in range(10):
            arch = self.space.index_to_arch(idx)
            assert arch.cell_ops is not None
            assert len(arch.layer_widths) >= 3  # at least input + 1 hidden + output
