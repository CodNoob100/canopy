# Reproducible NAS experiments and provenance

CANOPY has **two separate evaluation interfaces**, with different output schemas and different evidence guarantees. Neither makes synthetic accuracies into real measurements:

| Interface | Data source | Output |
| --- | --- | --- |
| [`ExperimentRunner.run_all()`](../canopy/evaluation/runner.py) | Passed-in benchmark object (real, synthetic or custom) | Results JSON, trajectory CSV, **experiment manifest JSON** |
| [`canopy evaluate`](cli.md) | Explicit **real NATS-Bench TSS** path only | `benchmark_comparison.json` with aggregate metrics and sampled indices; **no runner manifest** |

## ExperimentRunner artifacts

Each `run_all(n_seeds=...)` invocation writes three files in its configured output directory:

- `search_results_<run-id>.json`: per-seed strategies, final scores/accuracies, trajectories and missing-data indicators.
- `search_trajectory_<run-id>.csv`: per-evaluation best-observed accuracy with `accuracy_source`, `benchmark_lookup_available` and `fitness_score_units` columns.
- `experiment_manifest_<run-id>.json`: schema version, CANOPY/runtime/platform and dependency versions, Git commit **when available**, search-space class, strategies, seed list, budget, score interpretations and output filenames.

The manifest's `files` map points to the **results JSON and trajectory CSV**. The generated timestamp/file names and wall-clock duration are **not deterministic**. A manifest is a record of configuration, not a certificate that an external benchmark or its checksum was trustworthy.

## Small offline reproducibility example (simulation only)

```python
from canopy.evaluation.runner import ExperimentRunner
from canopy.nas.architecture import NASBench201SearchSpace
from canopy.nas.benchmark import SyntheticBenchmark

runner = ExperimentRunner(
    search_space=NASBench201SearchSpace(),
    benchmark=SyntheticBenchmark(n_architectures=50, seed=42),
    strategies=["tropical_generic", "random"],
    population_size=8, n_generations=3, max_evaluations=10,
    output_dir="results/offline-smoke",
)
runner.run_all(n_seeds=2)
```

This is a **software test/simulation**, not a benchmark accuracy experiment: `SyntheticBenchmark` derives simulated accuracies partly from the scoring function, creating circular target labels for any purported validation of score/accuracy correlation.

Random-strategy fitness values use an isolated `numpy.random.default_rng(seed)`. A fixed seed can reproduce ranking/search paths for a fixed code revision, search space, benchmark content, dependency environment and deterministic backend; it cannot make run duration or output filenames deterministic. The DARTS example also preserves the seeded fitness hook.

## Real, simulated and unknown measurements

- `SyntheticBenchmark` data are classified `synthetic_simulation`; never call these results ground-truth accuracy.
- The NATS-Bench API wrapper is classified `nats_bench_reported` when supplied, but **code identity alone does not authenticate** the caller's actual benchmark file.
- A custom benchmark object is classified `unclassified_benchmark` until its data provenance is established.
- A missing benchmark entry **or an entry whose test accuracy is unavailable** is recorded as JSON `null` rather than a fictitious `0.0`; a genuine measured `0.0` is retained. Non-finite or out-of-range accuracy values are rejected.
- The legacy trajectory CSV column `best_test_accuracy` reflects the **best available observed** value so far and may be blank when none was observed. Read `accuracy_source` and `benchmark_lookup_available` before interpreting it.
- `missing_benchmark_lookups` and `trajectory_lookup_available` describe data coverage; they are not accuracy-quality metrics.

Scoring units are **not interchangeable**: `tropical_generic` and `tropical_tight` use `count_bound`; the random baseline is `unit_interval_random`; parameter count uses `benchmark_parameter_score` (benchmark-specific units). The standalone DARTS structural proxy instead returns an approximate **log10-scale** score.

## Benchmark identifiers and digest semantics

With a real `NATSBenchAPI` or custom benchmark, `ExperimentRunner(..., benchmark_source_id="...", benchmark_sha256="<64-hex-digest>")` stores metadata. **ExperimentRunner does not independently compute or verify that SHA-256**. The manifest uses `user_supplied_not_verified` to prevent falsely certifying it; do not include secrets or sensitive local paths in identifiers.

By contrast, the separate `canopy evaluate --sha256` option **computes and compares** a digest against one explicitly supplied benchmark **file**, not a directory. That only verifies equality to a caller-supplied digest: provenance requires an independently trusted source for the expected hash. NATS-Bench data may use unsafe serialized formats; never deserialize untrusted files.

## Reproduction checklist

1. Pin the repository commit, Python version and installed package versions, including any separately installed NAS-Bench-301 packages.
2. Document whether the benchmark is **official real NATS-Bench TSS**, another independently measured dataset, a NAS-Bench-301 **surrogate prediction**, or simulated/unknown data.
3. Record publisher, benchmark artifact identifier, release and trusted digest provenance (where available), dataset (`cifar10`, `cifar100` or `ImageNet16-120`), benchmark hyperparameter setting (`hp`), sampled architecture indices and seeds.
4. Keep the scoring method, score units, baselines, budget, sampling design, missing-data counts, environment and comparison metric definitions together with machine-readable outputs.
5. Repeat a fixed-seed run in an equivalent environment and check raw data/proxy scores. Do not infer generalization from a synthetic dataset correlated with the score by construction.
6. Treat historical results plots and DARTS numerical claims as **unverified** until an auditable, real-data/model experiment establishes them. See [research evidence and limitations](research-evidence.md) and the pending [real TSS integration issue #29](https://github.com/CodNoob100/canopy/issues/29).

**Known limitation:** `canopy evaluate` currently writes aggregate comparisons and sample indices, not a complete per-architecture raw results table. For an independently auditable publication-grade evaluation, separately retain/release appropriately licensed raw measurements and scoring results; a checksum and a passing CI badge alone are insufficient.
