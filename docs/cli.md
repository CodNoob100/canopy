# CANOPY scoring CLI

The `canopy` command is installed with CANOPY (`pip install -e .`).

```bash
canopy score --widths 2,64,64,1 --mode generic
canopy score --widths 2,64,64,1 --mode tight --json
python -m canopy.cli score --widths 2,4,1 --json
```

**Width order:** input dimension, all hidden-layer widths, then output
dimension. Every width must be a positive integer.

The `generic` and `tight` modes are architecture-level combinatorial
**upper bounds**, not exact counts of the regions formed by particular
network weights. The CLI describes the score as `count_bound` rather than
mislabeling it as a log10 count.

`--json` outputs version, widths, mode, score units, and interpretation.
The score is a **decimal string** in JSON, avoiding precision loss with very
large combinatorial integers in downstream JavaScript and JSON consumers.

This feature scores feedforward ReLU architectures only. Evaluation against
real NAS-Bench datasets, DARTS proxy scoring, and experiment manifests remain
distinct future interfaces. Do not interpret synthetic accuracies as real
benchmark validation.
