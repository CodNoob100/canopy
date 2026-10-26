# CANOPY command-line interface

The `canopy` console script is installed from [`pyproject.toml`](../pyproject.toml). Its two subcommands do different things: `score` computes feedforward architecture-level bounds **without training**, while `evaluate` compares scores against an **explicitly provided** NATS-Bench topology-search-space (TSS) benchmark.

## Installation

```bash
python -m pip install -e .
# Additionally needed for real TSS benchmark access
python -m pip install -e '.[bench]'
```

The first command installs NumPy and SciPy. The optional `bench` extra installs the `nats-bench` API, **not** its external benchmark data. The `eval` extra adds plotting/analysis libraries for example scripts, and does not change the CLI's benchmark trust policy.

## `canopy score`: feedforward architecture-only bound

```bash
canopy score --widths 2,64,64,1 --mode generic
canopy score --widths 2,64,64,1 --mode tight --json
python -m canopy.cli score --widths 2,4,1 --json
```

Layer widths are **input, zero or more hidden layers, output**; every width must be a positive integer. Supported modes:

- `generic`: layer-wise product of hyperplane-arrangement factors.
- `tight`: a bottleneck-aware combinatorial upper bound.

Both return **integer architecture-level upper bounds** (`count_bound`), not log10 values and not counts realized by a particular set of trained/initialized weights. No DARTS operations or benchmark data are accepted by this command. See [`generic_linear_region_count`](../canopy/math/tropical_theory.py) and [`tropical_expressivity_score`](../canopy/nas/tropical_score.py).

`--json` includes `canopy_version`, `layer_widths`, `mode`, `score`, `score_units` and `score_interpretation`. The potentially large integer `score` is a **decimal string** to avoid JavaScript/JSON precision loss.

## `canopy evaluate`: explicitly supplied real NATS-Bench TSS data

Get a trusted **TSS/topology** artifact from the [official NATS-Bench repository](https://github.com/D-X-Y/NATS-Bench). Use either its extracted `NATS-tss-v1_0-3ffb9-simple` directory or the matching `NATS-tss-v1_0-3ffb9.pickle.pbz2` file. The CLI does not download the data and never silently generates synthetic accuracies.

```bash
canopy evaluate \
  --benchmark-data /trusted/path/to/NATS-tss-v1_0-3ffb9-simple \
  --dataset cifar100 --hp 200 --seed 42 --limit 200 \
  --output results/cifar100-seed42
```

| Option | Behavior |
| --- | --- |
| `--benchmark-data` | **Required.** Path to a trusted real TSS file or directory; not a SSS/size-space benchmark |
| `--dataset` | **Required.** `cifar10`, `cifar100`, or `ImageNet16-120` |
| `--seed` | **Required.** Integer used to sample architecture indices without replacement |
| `--output` | **Required.** A **new** output directory; existing paths are rejected |
| `--hp` | `12` or `200`; defaults to `200` |
| `--limit` | Number of distinct architectures, **2–1,000**; defaults to `200` |
| `--benchmark-id` | Optional caller-provided provenance label |
| `--sha256` | Optional **trusted external** 64-character SHA-256 expectation; supported only when benchmark data path is a **file** |

Failure to find the artifact, invalid or missing test accuracies, missing measured parameters/FLOPs, mismatched sample indices, or failed API lookups raises an error rather than inventing a replacement value. The comparison is retrospective: it does **not** train architectures or substantiate earlier DARTS research claims.

### Output and limitations

The command writes `benchmark_comparison.json` in the new output directory. The JSON records CANOPY version, benchmark `kind`, TSS dataset/`hp`, provenance label and checksum **status**, seed, the complete sampled index list, missing lookup count, and **aggregate** comparisons (`spearman_rho`, tie-aware `kendall_tau`, `top_10_accuracy`, `top_50_accuracy`) for the generic/tight bounds, **measured** parameter count and FLOPs, and a random baseline.

Undefined correlations are stored as JSON `null`, not `0`; top-K overlap is `null` if the sampled count is **K or smaller**, because selecting everything is uninformative. The CLI report is **not** a full per-architecture raw-score/accuracy table and does **not** write an `ExperimentRunner` manifest. Preserve separate raw data and environment provenance when conducting publishable research.

Providing a SHA-256 merely verifies bytes against the **caller-supplied** digest. It does not independently authenticate its publisher. Files may be stored in formats that deserialize objects; **never load untrusted benchmark archives**, even when a checksum matches an untrusted source.

### Verification status

The implementation and invalid-data paths are covered by [`tests/test_evaluate_cli.py`](../tests/test_evaluate_cli.py) with offline fixtures and injected benchmark backends. Python CI passing does **not** demonstrate that this command was executed against a verified genuine NATS-Bench archive. This remains a tracked research validation step: [issue #29](https://github.com/CodNoob100/canopy/issues/29).

## Related research interfaces

`tropical_expressivity_score_darts(arch)` is a **log10-scale DARTS structural proxy**, not a `score` CLI mode. The `ExperimentRunner` creates its own results JSON, trajectory CSV and experiment manifest; see [reproducibility documentation](reproducibility.md). Model-based DARTS scripts require a separately installed [NAS-Bench-301 surrogate](https://github.com/automl/nasbench301) and trusted models, and those predictions are not direct measured training outcomes. See the [evidence ledger](research-evidence.md).
