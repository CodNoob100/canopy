# The Motivation Behind CANOPY

Neural Architecture Search (NAS) promises to automate the design of deep learning models. However, training thousands of candidate architectures to convergence to find the best one requires massive, often prohibitive, computational resources.

To solve this, the community turned to **Training-Free (Zero-Cost) Proxies**. These are fast heuristics-like parameter count, FLOPs, gradient flow (Synflow), or Jacobian covariance (Jacov/NASWOT)-that can score an architecture's potential at initialization, without training any weights.

### The Problem with Existing Proxies

While existing zero-cost proxies correlate empirically with trained performance in constrained settings, they all share a fundamental weakness: **they lack a direct mathematical connection to representational capacity.** 

- **The Parameter Count Failure**: In DARTS, many architectures with identical parameter counts - including skip-heavy topologies with low actual expressivity - are ranked equally by parameter-based metrics. A proxy based purely on parameter count cannot differentiate between a useful wide convolution and a redundant dense skip connection.

### The Solution: Tropical Geometry

The question we asked was: *Is there a training-free NAS metric that measures expressivity directly, with a mathematical proof connecting the score to the network's representational capacity?*

The answer lies in **Tropical Geometry**. A foundational theorem (Zhang et al. 2018; Maragos et al. 2021) establishes an exact mathematical equivalence: **every feedforward ReLU network computes exactly a tropical rational function**. 

This means that the number of distinct linear regions a ReLU network can express (its true expressivity) is exactly equal to the number of maximal cells in the corresponding tropical hypersurface. 

### CANOPY: Expressivity as a Metric

CANOPY is the first training-free NAS metric built entirely on this tropical equivalence. Instead of guessing performance using gradients or parameter counts, CANOPY calculates the architecture's exact theoretical expressivity.

To make this work on real-world NAS benchmarks, CANOPY introduces critical mathematical extensions to the tropical framework:
1. **Convolutional Extension**: We use Zaslavsky's theorem on Toeplitz-structured hyperplane arrangements to map spatial weight sharing.
2. **Residual Skip Connections**: We model the addition of identity pathways via the Minkowski sum of Newton polytopes and normal fan refinements.

### Key Finding: The Bound Tightness Paradox

While building CANOPY, we discovered a highly counter-intuitive phenomenon we call the **Bound Tightness Paradox**. 

When calculating the expressivity bounds of neural networks, mathematicians strive for "tighter" bounds-formulas that get closer to the exact number of linear regions. However, we found empirically that **mathematically tighter bounds produce worse NAS rankings**.

Why? Because the "looseness" of the generic bounds (like the classic Montúfar bound or the Cartesian product bound for skip connections) actually acts as a high-fidelity proxy for network density. In search spaces where density correlates with accuracy, this mathematical "error" becomes statistically beneficial. 

This paradox fundamentally shifts how we think about NAS metric design: we should evaluate expressivity proxies based on their ranking quality, not just their mathematical tightness.
