# Why CANOPY studies tropical geometry for neural architecture search

Neural architecture search (NAS) explores network designs, but fully training every candidate can be expensive. **Training-free proxies** offer inexpensive rankings before full training. Common baselines include parameter count, FLOPs and measures derived from gradients or activations. These scores can be useful, but no proxy is universally guaranteed to predict held-out accuracy.

## Mathematical motivation, not an exact architecture-level measurement

Piecewise-linear networks such as ReLU networks admit mathematical descriptions related to tropical algebra and geometry. [Zhang, Naitzat and Lim (ICML 2018)](https://proceedings.mlr.press/v80/zhang18i.html) establish a relationship between classes of ReLU networks and tropical rational maps under the paper's conditions and relate linear regions to associated polyhedral structures. [Montúfar et al. (NeurIPS 2014)](https://papers.nips.cc/paper_files/paper/2014/hash/fa6f2a469cc4d61a92d96e74617c3d2a-Abstract.html) study theoretical linear-region complexity. These mathematical results **do not imply** that layer widths alone determine the exact number of linear regions formed by a particular trained network.

CANOPY uses that motivation to explore fast architectural scoring:

- **Feedforward architecture-only bounds.** [`generic_linear_region_count`](canopy/math/tropical_theory.py) combines Zaslavsky-style hyperplane-arrangement factors by layer width; [`linear_region_upper_bound_tight`](canopy/math/tropical_theory.py) introduces bottleneck awareness. Both return combinatorial integer **upper bounds**, not exact realized counts.
- **Weight-aware region counting.** [`count_linear_regions`](canopy/geometry/cell_counter.py) includes algorithms selected by network size and method. Results may be exact or estimated; they depend on instantiated weights and the chosen method.
- **Convolutional and skip-connection proxies.** [`tropical_expressivity_score_darts`](canopy/nas/tropical_score.py) uses simplifying structural assumptions for DARTS-style operations. It reports a **log10-scale heuristic proxy**, not an exact maximum over convolution filters or a validated measure of true network expressivity.

## Research questions

The project explores whether theoretically motivated architectural features correlate with independent benchmark outcomes, whether they add information beyond parameter/FLOPs baselines, and whether tighter combinatorial bounds improve rankings. The answers are **empirical questions**, not consequences of the tropical representation theorems.

### The historical 'bound-tightness paradox' hypothesis

Earlier project text and figures suggest that some looser structural bounds may produce better rankings than some tighter ones because they indirectly encode architecture density. This is a **historical project hypothesis**, not a general mathematical theorem or a demonstrated, independently replicated empirical conclusion. A meaningful test must define the compared bounds, datasets, split/sample selection, seeds, uncertainty and independent baselines.

Historical visualizations are included in [`results/`](results/). The previously reported DARTS Spearman rho figures (0.51 versus 0.33) are **not yet tied to a fully verified, reproducible external-data/model run** in this repository. They should not be described as confirmed evidence of superiority.

## How to evaluate the hypothesis responsibly

Use [the real-data evaluation CLI](docs/cli.md) with a trusted NATS-Bench TSS artifact, record the sampled indices, benchmark identity, versioned environment, score units, seeds and raw outputs, and evaluate rank metrics with appropriate handling of ties and undefined cases. Treat synthetic benchmark accuracies as simulation, never as independent target labels: the simulator itself can depend on the tropical score.

For DARTS/NAS-Bench-301, use separately installed, appropriately versioned surrogate models; surrogate-predicted accuracy is **not** accuracy measured by retraining each candidate. Record the model version, configuration, sampling procedure and all baseline assumptions before making claims.

See the [README](README.md) for the user-facing API summary, the [evidence ledger](docs/research-evidence.md) for known verification limits, and [issue #29](https://github.com/CodNoob100/canopy/issues/29) for pending trusted-real-data integration validation.

**Scope:** CANOPY is an exploratory research framework. The current tested implementation is not evidence, by itself, that a particular architecture score will select a better trained model.
