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

## Real NATS-Bench evaluation

Use the evaluation command to compare architecture-level scoring proxies
with **real, explicitly provided** NATS-Bench TSS/NAS-Bench-201 entries:

```bash
pip install -e '.[bench]'
canopy evaluate \\
  --benchmark-data /trusted/path/to/NATS-tss-v1_0-3ffb9-simple \\
  --dataset cifar100 --hp 200 --seed 42 --limit 200 \\
  --output results/cifar100-seed42
```

`--benchmark-data`, `--dataset`, `--seed`, and `--output` are mandatory.
No synthetic fallback is allowed; missing or invalid architecture records
cause an error instead of being omitted. The command samples indices
uniformly without replacement and records the exact chosen indices.
Limits are 2–1,000 architectures to bound the quadratic Kendall calculation.

For a **single trusted benchmark file**, `--sha256 VERIFIED_DIGEST` verifies
a caller-supplied 64-digit hexadecimal checksum before loading any data.
The tool does not verify the provenance of the checksum itself and will
label data without a checksum as unverified. NATS-Bench files may use
unsafe serialization formats; do not load archives from untrusted sources.

The report is written to a **new** output directory as
`benchmark_comparison.json` and contains the benchmark metadata,
sample seed and indices, and Spearman/Kendall/top-K comparisons.
Undefined correlations appear as JSON null, not fabricated zeros.

This retrospective test-accuracy evaluation does not prove CANOPY's
NAS-Bench-301 claims or constitute training a neural architecture. It
is independent of ExperimentRunner's separately proposed manifests.
