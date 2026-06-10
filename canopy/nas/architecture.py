"""
Architecture Specification and Search Space
=============================================

Defines the architecture encoding used throughout the NAS pipeline,
plus search-space definitions compatible with NAS-Bench-201.

Design decisions
----------------
- ArchitectureSpec is deliberately lightweight (a frozen dataclass)
  so it can be hashed, cached, and compared cheaply.
- The encoding/decoding between flat integer vectors and ArchitectureSpec
  enables gradient-free search operators (mutation, crossover).
- NASBench201SearchSpace mirrors the 4-node, 6-edge cell structure of
  NAS-Bench-201 for direct benchmark comparison.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple


# ---------------------------------------------------------------------------
#  ArchitectureSpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArchitectureSpec:
    """
    Immutable specification of a feedforward ReLU network architecture.

    This is the unit of search in the NAS loop.  It describes the
    *structure* of a network — not its weights.

    Attributes
    ----------
    layer_widths : tuple of int
        (n_0, n_1, …, n_L) where n_0 = input dim, n_L = output dim,
        and n_1…n_{L-1} are hidden-layer widths.
    depth : int
        Number of layers L (= len(layer_widths) - 1).
    connectivity : tuple of tuple of int, optional
        Adjacency list encoding skip connections.
        connectivity[l] lists the layers whose output feeds into layer l
        (in addition to the default layer l-1).
    activation : str
        Activation function name.  Only 'relu' is supported for
        tropical analysis.
    cell_ops : tuple of str, optional
        For cell-based search spaces (NAS-Bench-201), the operation
        assigned to each edge.
    """

    layer_widths: Tuple[int, ...]
    depth: int = 0
    connectivity: Optional[Tuple[Tuple[int, ...], ...]] = None
    activation: str = "relu"
    cell_ops: Optional[Tuple[str, ...]] = None

    def __post_init__(self):
        if self.depth == 0:
            object.__setattr__(self, "depth", len(self.layer_widths) - 1)
        if self.activation != "relu":
            raise ValueError(
                f"Only 'relu' is supported for tropical analysis, "
                f"got '{self.activation}'"
            )

    @property
    def input_dim(self) -> int:
        return self.layer_widths[0]

    @property
    def output_dim(self) -> int:
        return self.layer_widths[-1]

    @property
    def hidden_widths(self) -> Tuple[int, ...]:
        return self.layer_widths[1:-1]

    @property
    def total_hidden_neurons(self) -> int:
        return sum(self.hidden_widths)

    @property
    def total_params(self) -> int:
        """Approximate parameter count (weights + biases)."""
        count = 0
        for i in range(len(self.layer_widths) - 1):
            n_in = self.layer_widths[i]
            n_out = self.layer_widths[i + 1]
            count += n_out * n_in + n_out  # weights + bias
        return count

    def to_vector(self) -> np.ndarray:
        """Encode as a flat integer vector (for search operators)."""
        parts = list(self.layer_widths)
        if self.cell_ops is not None:
            parts.extend(
                [NASBench201SearchSpace.OPS.index(op) for op in self.cell_ops]
            )
        return np.array(parts, dtype=int)

    @staticmethod
    def from_vector(
        vec: np.ndarray,
        has_cell_ops: bool = False,
        n_cell_edges: int = 6,
    ) -> ArchitectureSpec:
        """Decode from a flat integer vector."""
        vec = np.asarray(vec, dtype=int)
        if has_cell_ops:
            widths = tuple(vec[:-n_cell_edges])
            op_indices = vec[-n_cell_edges:]
            ops = tuple(NASBench201SearchSpace.OPS[i] for i in op_indices)
            return ArchitectureSpec(layer_widths=widths, cell_ops=ops)
        return ArchitectureSpec(layer_widths=tuple(vec))

    def __repr__(self) -> str:
        s = f"ArchitectureSpec(widths={self.layer_widths}"
        if self.cell_ops:
            s += f", ops={self.cell_ops}"
        s += ")"
        return s


# ---------------------------------------------------------------------------
#  Search Spaces
# ---------------------------------------------------------------------------

class SearchSpace:
    """
    Base class for architecture search spaces.

    A search space defines:
    - How to sample a random architecture.
    - How to mutate an existing architecture.
    - How to crossover two architectures.
    - Bounds on layer widths and depth.
    """

    def random(self, rng: np.random.Generator) -> ArchitectureSpec:
        raise NotImplementedError

    def mutate(
        self, arch: ArchitectureSpec, rng: np.random.Generator
    ) -> ArchitectureSpec:
        raise NotImplementedError

    def crossover(
        self,
        parent_a: ArchitectureSpec,
        parent_b: ArchitectureSpec,
        rng: np.random.Generator,
    ) -> ArchitectureSpec:
        raise NotImplementedError


class FeedforwardSearchSpace(SearchSpace):
    """
    Search space over feedforward architectures with variable depth
    and width.

    Parameters
    ----------
    input_dim : int
        Fixed input dimension.
    output_dim : int
        Fixed output dimension.
    min_depth : int
        Minimum number of hidden layers.
    max_depth : int
        Maximum number of hidden layers.
    min_width : int
        Minimum hidden-layer width.
    max_width : int
        Maximum hidden-layer width.
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 1,
        min_depth: int = 1,
        max_depth: int = 5,
        min_width: int = 2,
        max_width: int = 64,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_width = min_width
        self.max_width = max_width

    def random(self, rng: np.random.Generator) -> ArchitectureSpec:
        depth = rng.integers(self.min_depth, self.max_depth + 1)
        widths = rng.integers(
            self.min_width, self.max_width + 1, size=depth
        ).tolist()
        layer_widths = tuple(
            [self.input_dim] + widths + [self.output_dim]
        )
        return ArchitectureSpec(layer_widths=layer_widths)

    def mutate(
        self, arch: ArchitectureSpec, rng: np.random.Generator
    ) -> ArchitectureSpec:
        """
        Mutate an architecture by randomly:
        1. Changing one hidden-layer width (±1 to ±8),
        2. Adding a hidden layer, or
        3. Removing a hidden layer.
        """
        hidden = list(arch.hidden_widths)
        action = rng.choice(["width", "add", "remove"], p=[0.6, 0.2, 0.2])

        if action == "width" and len(hidden) > 0:
            idx = rng.integers(len(hidden))
            delta = rng.integers(-8, 9)
            hidden[idx] = max(self.min_width, min(self.max_width, hidden[idx] + delta))

        elif action == "add" and len(hidden) < self.max_depth:
            idx = rng.integers(len(hidden) + 1)
            w = rng.integers(self.min_width, self.max_width + 1)
            hidden.insert(idx, w)

        elif action == "remove" and len(hidden) > self.min_depth:
            idx = rng.integers(len(hidden))
            hidden.pop(idx)

        else:
            # Fallback: width mutation
            if len(hidden) > 0:
                idx = rng.integers(len(hidden))
                delta = rng.integers(-4, 5)
                hidden[idx] = max(
                    self.min_width, min(self.max_width, hidden[idx] + delta)
                )

        layer_widths = tuple(
            [arch.input_dim] + hidden + [arch.output_dim]
        )
        return ArchitectureSpec(layer_widths=layer_widths)

    def crossover(
        self,
        parent_a: ArchitectureSpec,
        parent_b: ArchitectureSpec,
        rng: np.random.Generator,
    ) -> ArchitectureSpec:
        """
        Uniform crossover: for each layer position, randomly pick from
        parent_a or parent_b.  Uses the shorter parent's depth.
        """
        ha = list(parent_a.hidden_widths)
        hb = list(parent_b.hidden_widths)
        min_len = min(len(ha), len(hb))
        max_len = max(len(ha), len(hb))

        # Choose a depth in [min_len, max_len]
        depth = rng.integers(min_len, max_len + 1) if max_len > 0 else 0

        child: List[int] = []
        for i in range(depth):
            wa = ha[i] if i < len(ha) else ha[-1] if ha else self.min_width
            wb = hb[i] if i < len(hb) else hb[-1] if hb else self.min_width
            child.append(wa if rng.random() < 0.5 else wb)

        layer_widths = tuple(
            [parent_a.input_dim] + child + [parent_a.output_dim]
        )
        return ArchitectureSpec(layer_widths=layer_widths)


class NASBench201SearchSpace(SearchSpace):
    """
    NAS-Bench-201 cell-based search space.

    Architecture: a cell with 4 nodes and 6 edges.  Each edge is assigned
    one of 5 operations.  The cell is stacked with fixed channel widths.

    The 5 operations:
        0: none (zero)
        1: skip_connect (identity)
        2: nor_conv_1x1
        3: nor_conv_3x3
        4: avg_pool_3x3

    The 6 edges connect nodes:
        (0→1), (0→2), (1→2), (0→3), (1→3), (2→3)

    Total search space: 5^6 = 15,625 architectures.

    For tropical expressivity scoring, each operation is mapped to an
    effective layer width:
        none        → 0 (no connection)
        skip_connect → n (pass-through, width = input)
        nor_conv_1x1 → n (1×1 conv preserves channels)
        nor_conv_3x3 → n (3×3 conv preserves channels)
        avg_pool_3x3 → n (pooling preserves channels)
    """

    OPS = ("none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3")
    NUM_OPS = 5
    NUM_EDGES = 6
    EDGE_CONNECTIONS = (
        (0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)
    )
    TOTAL_ARCHITECTURES = NUM_OPS ** NUM_EDGES  # 15,625

    # Effective expressivity multiplier per operation (heuristic)
    # Higher = more expressive.  None = zero contribution.
    OP_EXPRESSIVITY = {
        "none": 0.0,
        "skip_connect": 0.5,    # pass-through, no new features
        "nor_conv_1x1": 0.8,    # mixing, limited spatial
        "nor_conv_3x3": 1.0,    # full convolution
        "avg_pool_3x3": 0.3,    # dimensionality reduction
    }

    def __init__(self, channels: int = 16, num_classes: int = 10):
        self.channels = channels
        self.num_classes = num_classes

    def random(self, rng: np.random.Generator) -> ArchitectureSpec:
        ops = tuple(
            self.OPS[rng.integers(self.NUM_OPS)]
            for _ in range(self.NUM_EDGES)
        )
        return self._ops_to_arch(ops)

    def mutate(
        self, arch: ArchitectureSpec, rng: np.random.Generator
    ) -> ArchitectureSpec:
        """Mutate one random edge's operation."""
        if arch.cell_ops is None:
            raise ValueError("Architecture has no cell_ops for NAS-Bench-201")
        ops = list(arch.cell_ops)
        edge = rng.integers(self.NUM_EDGES)
        new_op = self.OPS[rng.integers(self.NUM_OPS)]
        ops[edge] = new_op
        return self._ops_to_arch(tuple(ops))

    def crossover(
        self,
        parent_a: ArchitectureSpec,
        parent_b: ArchitectureSpec,
        rng: np.random.Generator,
    ) -> ArchitectureSpec:
        """Uniform crossover over edge operations."""
        if parent_a.cell_ops is None or parent_b.cell_ops is None:
            raise ValueError("Both parents must have cell_ops")
        ops = tuple(
            parent_a.cell_ops[i] if rng.random() < 0.5 else parent_b.cell_ops[i]
            for i in range(self.NUM_EDGES)
        )
        return self._ops_to_arch(ops)

    def _ops_to_arch(self, ops: Tuple[str, ...]) -> ArchitectureSpec:
        """
        Convert cell operations to an ArchitectureSpec.

        Maps the cell structure to an equivalent feedforward representation
        for tropical scoring.  Each non-none edge contributes an effective
        "sub-layer" whose width is modulated by the operation type.
        """
        # Compute effective width at each node
        node_widths = [self.channels]  # node 0
        for target in range(1, 4):
            incoming = []
            for edge_idx, (src, dst) in enumerate(self.EDGE_CONNECTIONS):
                if dst == target:
                    op = ops[edge_idx]
                    mult = self.OP_EXPRESSIVITY[op]
                    if mult > 0:
                        src_w = node_widths[src] if src < len(node_widths) else self.channels
                        incoming.append(int(np.ceil(mult * src_w)))
            width = sum(incoming) if incoming else 0
            node_widths.append(max(width, 1))

        # Build layer_widths: input → cell layers → output
        hidden = [w for w in node_widths if w > 0]
        layer_widths = tuple(
            [self.channels] + hidden + [self.num_classes]
        )
        return ArchitectureSpec(layer_widths=layer_widths, cell_ops=ops)

    def enumerate_all(self) -> List[ArchitectureSpec]:
        """Enumerate all 15,625 architectures (for exhaustive evaluation)."""
        from itertools import product as iterproduct

        archs = []
        for ops_indices in iterproduct(range(self.NUM_OPS), repeat=self.NUM_EDGES):
            ops = tuple(self.OPS[i] for i in ops_indices)
            archs.append(self._ops_to_arch(ops))
        return archs

    def index_to_arch(self, index: int) -> ArchitectureSpec:
        """Convert a flat index (0..15624) to an architecture."""
        ops_indices = []
        remaining = index
        for _ in range(self.NUM_EDGES):
            ops_indices.append(remaining % self.NUM_OPS)
            remaining //= self.NUM_OPS
        ops = tuple(self.OPS[i] for i in ops_indices)
        return self._ops_to_arch(ops)

    def arch_to_index(self, arch: ArchitectureSpec) -> int:
        """Convert an architecture back to its flat index."""
        if arch.cell_ops is None:
            raise ValueError("Architecture has no cell_ops")
        index = 0
        for i, op in enumerate(arch.cell_ops):
            index += self.OPS.index(op) * (self.NUM_OPS ** i)
        return index


class DARTS_SearchSpace(SearchSpace):
    """
    NAS-Bench-301 DARTS search space.
    
    A DARTS cell has 4 intermediate nodes, and 2 input nodes.
    Each intermediate node must select exactly 2 incoming edges from previous nodes.
    We encode this as a tuple of 8 operations (2 per node), where each operation 
    is actually a tuple: (op_name, input_node_idx).
    """
    OPS = (
        "max_pool_3x3",
        "avg_pool_3x3",
        "skip_connect",
        "sep_conv_3x3",
        "sep_conv_5x5",
        "dil_conv_3x3",
        "dil_conv_5x5"
    )
    NUM_OPS = 7

    def random(self, rng: np.random.Generator) -> ArchitectureSpec:
        """Randomly sample a valid discrete DARTS architecture."""
        normal = []
        for i in range(4):
            choices = rng.choice(self.OPS, size=2, replace=True)
            nodes = rng.choice(range(i + 2), size=2, replace=False)
            normal.append(f"{choices[0]}~{nodes[0]}")
            normal.append(f"{choices[1]}~{nodes[1]}")
        return ArchitectureSpec(layer_widths=(), cell_ops=tuple(normal))

    def mutate(self, arch: ArchitectureSpec, rng: np.random.Generator) -> ArchitectureSpec:
        if arch.cell_ops is None:
            raise ValueError("Architecture has no cell_ops")
        ops = list(arch.cell_ops)
        edge_idx = rng.integers(8)
        node_idx = edge_idx // 2
        
        # Mutate either the operation or the connection
        if rng.random() < 0.5:
            # Change operation
            new_op = self.OPS[rng.integers(self.NUM_OPS)]
            curr_src = ops[edge_idx].split("~")[1]
            ops[edge_idx] = f"{new_op}~{curr_src}"
        else:
            # Change connection (must be different from the other edge of the same node)
            curr_op = ops[edge_idx].split("~")[0]
            other_edge_idx = edge_idx + 1 if edge_idx % 2 == 0 else edge_idx - 1
            other_src = int(ops[other_edge_idx].split("~")[1])
            valid_srcs = [s for s in range(node_idx + 2) if s != other_src]
            new_src = rng.choice(valid_srcs)
            ops[edge_idx] = f"{curr_op}~{new_src}"
            
        return ArchitectureSpec(layer_widths=(), cell_ops=tuple(ops))

    def crossover(self, parent_a: ArchitectureSpec, parent_b: ArchitectureSpec, rng: np.random.Generator) -> ArchitectureSpec:
        if parent_a.cell_ops is None or parent_b.cell_ops is None:
            raise ValueError("Both parents must have cell_ops")
        ops = []
        # Crossover per node (take both edges from A or both from B to avoid connection collisions)
        for i in range(4):
            if rng.random() < 0.5:
                ops.extend([parent_a.cell_ops[2*i], parent_a.cell_ops[2*i+1]])
            else:
                ops.extend([parent_b.cell_ops[2*i], parent_b.cell_ops[2*i+1]])
        return ArchitectureSpec(layer_widths=(), cell_ops=tuple(ops))

class TransformerSpec(ArchitectureSpec):
    """
    Architecture specification specifically tailored for Transformer models.
    Inherits from ArchitectureSpec, storing configuration in layer_widths 
    and cell_ops for backward compatibility with the NAS loop.
    """
    pass


class TransformerSearchSpace(SearchSpace):
    """
    Search space over Transformer architectures.
    Each layer is represented as an Attention block, encoded in `cell_ops`.
    """
    def __init__(self, min_depth=1, max_depth=6, min_d_model=64, max_d_model=512, allowed_heads=(2,4,8,16), N_seq=128):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_d_model = min_d_model
        self.max_d_model = max_d_model
        self.allowed_heads = allowed_heads
        self.N_seq = N_seq

    def random(self, rng: np.random.Generator) -> TransformerSpec:
        depth = rng.integers(self.min_depth, self.max_depth + 1)
        d_model = int(rng.integers(self.min_d_model, self.max_d_model + 1))
        layer_widths = [d_model] * (depth + 1)
        
        ops = []
        for _ in range(depth):
            valid_heads = [h for h in self.allowed_heads if d_model % h == 0]
            if not valid_heads: valid_heads = [1]
            h = rng.choice(valid_heads)
            ops.append(f"attn~{self.N_seq}~{h}")
            
        return TransformerSpec(layer_widths=tuple(layer_widths), cell_ops=tuple(ops))
        
    def mutate(self, arch: ArchitectureSpec, rng: np.random.Generator) -> TransformerSpec:
        ops = list(arch.cell_ops)
        d_model = arch.layer_widths[0]
        
        action = rng.choice(["depth", "d_model", "heads"])
        if action == "depth":
            if rng.random() < 0.5 and len(ops) < self.max_depth:
                h = int(ops[-1].split("~")[2])
                ops.append(f"attn~{self.N_seq}~{h}")
            elif len(ops) > self.min_depth:
                ops.pop()
        elif action == "d_model":
            delta = rng.integers(-32, 33)
            d_model = max(self.min_d_model, min(self.max_d_model, d_model + delta))
            for i in range(len(ops)):
                h = int(ops[i].split("~")[2])
                if d_model % h != 0:
                    valid_heads = [vh for vh in self.allowed_heads if d_model % vh == 0]
                    new_h = rng.choice(valid_heads) if valid_heads else 1
                    ops[i] = f"attn~{self.N_seq}~{new_h}"
        else:
            idx = rng.integers(len(ops))
            valid_heads = [vh for vh in self.allowed_heads if d_model % vh == 0]
            if valid_heads:
                new_h = rng.choice(valid_heads)
                ops[idx] = f"attn~{self.N_seq}~{new_h}"
                
        layer_widths = tuple([d_model] * (len(ops) + 1))
        return TransformerSpec(layer_widths=layer_widths, cell_ops=tuple(ops))

    def crossover(self, parent_a: ArchitectureSpec, parent_b: ArchitectureSpec, rng: np.random.Generator) -> TransformerSpec:
        d_model = parent_a.layer_widths[0] if rng.random() < 0.5 else parent_b.layer_widths[0]
        ops = []
        min_depth = min(len(parent_a.cell_ops), len(parent_b.cell_ops))
        max_depth = max(len(parent_a.cell_ops), len(parent_b.cell_ops))
        depth = rng.integers(min_depth, max_depth + 1) if max_depth > min_depth else min_depth
        
        for i in range(depth):
            op_a = parent_a.cell_ops[i] if i < len(parent_a.cell_ops) else parent_a.cell_ops[-1]
            op_b = parent_b.cell_ops[i] if i < len(parent_b.cell_ops) else parent_b.cell_ops[-1]
            chosen_op = op_a if rng.random() < 0.5 else op_b
            h = int(chosen_op.split("~")[2])
            if d_model % h != 0:
                valid_heads = [vh for vh in self.allowed_heads if d_model % vh == 0]
                h = rng.choice(valid_heads) if valid_heads else 1
            ops.append(f"attn~{self.N_seq}~{h}")
            
        layer_widths = tuple([d_model] * (depth + 1))
        return TransformerSpec(layer_widths=layer_widths, cell_ops=tuple(ops))

