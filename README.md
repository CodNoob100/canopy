# CANOPY: Expressivity-Guided Neural Architecture Search via Tropical Geometry

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

CANOPY is a training-free Neural Architecture Search (NAS) framework that rigorously grounds zero-cost proxy scoring in tropical geometry. 

Existing zero-cost proxies (like parameter count or kernel matrix rank) correlate with test accuracy empirically but lack theoretical justification for *why* they measure capacity. CANOPY is derived from the exact mathematical equivalence between ReLU networks and **tropical rational functions**. 

CANOPY scores an architecture by calculating the number of maximal cells in its corresponding tropical hypersurface - directly quantifying the exact number of distinct linear regions (expressivity) the architecture can produce.

On NAS-Bench-301 (a search space of $10^{18}$ DARTS architectures), CANOPY achieves a Spearman rank correlation of $\rho = 0.51$, a 56% relative improvement over standard parameter count ($\rho = 0.33$).

## Key Results

CANOPY outperforms baseline proxies by correctly capturing the architectural expressivity, and scales to complex search spaces like DARTS where traditional metrics fail.

| DARTS Rank Correlation | Evolutionary Search Trajectory (DARTS) |
|:---:|:---:|
| ![DARTS Correlation](results/darts_correlation.png) | ![DARTS Search Trajectory](results/darts_search_trajectory.png) |

| NATS-Bench CIFAR-100 Correlation | The Bound Tightness Paradox |
|:---:|:---:|
| ![CIFAR-100 Correlation](results/rank_correlation_cifar100.png) | ![Paradox Mechanism](results/paradox_mechanism.png) |

## Installation

CANOPY requires Python 3.10+ and PyTorch.

```bash
git clone https://github.com/CodNoob100/canopy.git
cd canopy
pip install -e .
```

To run the benchmarking examples, you will also need the `nats-bench` dependency:
```bash
pip install -e .[bench]
```

## Quickstart

Score a simple feedforward architecture's expressivity directly from its layer widths, with no weights instantiated:

```python
from canopy.nas.architecture import ArchitectureSpec
from canopy.nas.tropical_score import tropical_expressivity_score

# Define an architecture with 3 hidden layers of widths [64, 128, 64]
arch = ArchitectureSpec(layer_widths=(64, 128, 64))

# Calculate the exact tropical expressivity score (log10 of the number of linear regions)
score = tropical_expressivity_score(arch, mode="generic")
print(f"Log10 Linear Regions: {score:.2f}")
```

Score a DARTS architecture (with convolutional and skip-connection branches) using the Toeplitz arrangement and Minkowski sum proxy bounds:

```python
from canopy.nas.architecture import ArchitectureSpec
from canopy.nas.tropical_score import tropical_expressivity_score_darts

# Define a DARTS cell operation sequence
# e.g. Node 1 gets sep_conv_3x3 from Node 0, Node 2 gets skip_connect from Node 1, etc.
ops = (
    "sep_conv_3x3", "skip_connect", "max_pool_3x3", "none", 
    "sep_conv_5x5", "skip_connect", "none", "sep_conv_3x3", 
    "dil_conv_3x3", "skip_connect", "max_pool_3x3", "sep_conv_5x5", 
    "none", "none"
)

arch = ArchitectureSpec(layer_widths=[32, 32, 32, 32, 32], cell_ops=ops)
darts_score = tropical_expressivity_score_darts(arch)
print(f"DARTS Proxy Expressivity Score: {darts_score:.2f}")
```

## Running the Examples

You can reproduce the paper's NATS-Bench and DARTS trajectory evaluations using the provided scripts in the `examples/` directory.

### NATS-Bench Evaluation
```bash
# Set the NATS-Bench data path
export NATS_BENCH_TSS_PATH="/path/to/NATS-tss-v1_0-3ffb9-simple"

# Run the evolutionary search simulation
python examples/run_evaluation.py
```

### DARTS Pipeline Evaluation
```bash
# Run a 10-seed evolutionary search over the DARTS search space
python examples/run_darts_search_trajectory.py --seeds 10
```

## License
This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
