# Research claims, evidence and replication status

This ledger distinguishes **what the code implements**, **what repository CI actually checks**, and **what has not been reproduced with external research assets**. It is not a substitute for a research methods section or a peer-reviewed validation.

## Evidence tiers

| Claim or artifact | Available evidence | Current status |
| --- | --- | --- |
| `generic` / `tight` feedforward scores are combinatorial architecture-level bounds | [`canopy/math/tropical_theory.py`](../canopy/math/tropical_theory.py), [`canopy/nas/tropical_score.py`](../canopy/nas/tropical_score.py), unit tests | **Implemented and tested**; not counts for particular learned weights |
| DARTS structural scoring | [`tropical_expressivity_score_darts`](../canopy/nas/tropical_score.py), heuristic convolution/skip assumptions | **Implemented**; log10-scale proxy, not exact DARTS region enumeration |
| NATS-Bench TSS retrospective evaluation CLI | [`canopy/cli.py`](../canopy/cli.py), [`canopy/nas/benchmark.py`](../canopy/nas/benchmark.py), [`tests/test_evaluate_cli.py`](../tests/test_evaluate_cli.py) | **Offline-tested** using injected/fake benchmark adapters, not yet real-asset verified |
| Seeded NAS experiments and manifests | [`canopy/evaluation/runner.py`](../canopy/evaluation/runner.py), [`tests/test_reproducibility.py`](../tests/test_reproducibility.py) | **Offline-tested**; manifests document run settings, not dataset authenticity |
| Dependency compatibility | [`dependency-compatibility.yml`](../.github/workflows/dependency-compatibility.yml), passing [integration run](https://github.com/CodNoob100/canopy/actions/runs/40341785489) | **Smoke-tested** NumPy, pytest and pycddlib combinations; does not exhaust optional research integrations |
| Archived DARTS rho = 0.51 vs parameter baseline rho = 0.33 / 56% relative comparison | Historical README wording and checked-in image | **Historical, unverified**; no independently checked artifact, metrics dataset or run manifest tied to the figure |
| Bound-tightness paradox as a general empirical conclusion | Historical [`results/paradox_mechanism.png`](../results/paradox_mechanism.png) and project narrative | **Hypothesis / historical interpretation**, not established from the available replication evidence |

### What a passing workflow establishes

The merged `main` branch passed [Python CI run 40342306109](https://github.com/CodNoob100/canopy/actions/runs/40342306109) after PR #14. That is evidence the tested software operations passed in GitHub-hosted jobs. It **does not** verify published performance figures, data provenance, independently reproduced benchmark correlations or the behavior of every optional third-party model integration.

The [`results/`](../results/) directory currently contains PNG figures without an accompanying checked-in raw record table or traceable machine-readable run manifest. Do not infer numerical validation from those images alone. If the source runs are recovered, preserve raw metrics, seed lists, sample indices, versioned environment, model/dataset provenance, scoring units, and report-generation code together.

## Mathematical and measurement boundaries

- Feedforward `generic` and `tight` values are **integer bounds** derived from layer widths; they do not reveal how many linear regions a particular set of weights realizes.
- The DARTS proxy models operations through simplifying assumptions and reports a **log10-scale structural heuristic**. Its assumptions should be tested separately from correctness of the Python implementation.
- Weight-aware region counting can be exact or approximate depending on the chosen counting strategy and network size; this does not turn architecture-only bounds into exact realized counts.
- Parameters and FLOPs in the `canopy evaluate` rankings are taken from the **benchmark entries**. The DARTS example's parameter-count control is a hand-coded **proxy**, not measured model parameter count.
- Correlation estimates and rankings depend on the benchmark artifact, dataset, hyperparameter setting, sample selection, seed and handling of ties or missing values. A surrogate model prediction (NAS-Bench-301) is **not** directly measured accuracy from retraining an architecture.

## Trusted benchmark sources

- [Official NATS-Bench](https://github.com/D-X-Y/NATS-Bench) describes **TSS/topology search space** benchmark data and the CIFAR-10, CIFAR-100, and ImageNet16-120 datasets. The project's documented TSS artifact names include `NATS-tss-v1_0-3ffb9.pickle.pbz2` and `NATS-tss-v1_0-3ffb9-simple`.
- [Official NAS-Bench-301](https://github.com/automl/nasbench301) documents surrogate models in the DARTS search space. Using its models requires external installation and versioned model assets, which CANOPY does not bundle.

These upstream links establish **dataset/model identity**, not independent endorsement of CANOPY's numerical outcomes. Loading untrusted serialized datasets or model archives can be unsafe. Obtain artifacts from a trusted publisher and verify provenance and any available independently trusted digest before use.

## Minimal NATS-Bench validation protocol (not yet completed)

1. Obtain a trusted official TSS benchmark and record its source, version, release URL, file identity, independently checked checksum if available, and local software environment. Never commit benchmark archives.
2. Install the project with `python -m pip install -e '.[bench,eval]'`; record `git rev-parse HEAD`, Python and dependency versions.
3. Run `canopy evaluate --benchmark-data /trusted/NATS-tss-v1_0-3ffb9-simple --dataset cifar100 --hp 200 --seed 42 --limit 200 --output results/real-tss-cifar100-42`, using a **new** output folder.
4. Inspect `benchmark_comparison.json` for benchmark kind `real_nats_bench`, exact sampled indices, dataset, `hp`, valid measured parameters/FLOPs and no missing sample lookups; compare repeated runs with the same inputs.
5. Recheck other supported datasets and evaluate effect sizes, uncertainty, sampling variation and baseline design before writing scientific conclusions. Separately audit any NAS-Bench-301 surrogate-model-based DARTS outcomes.
6. Attach a reproducibility record and results to [issue #29](https://github.com/CodNoob100/canopy/issues/29) before promoting historical figures or correlations to validated claims.

The `canopy evaluate` CLI does **not** automatically emit an `ExperimentRunner` manifest. The CLI report records sample and benchmark metadata; add external environment and source-artifact provenance to reproduce an experiment.

## Change policy

New README or motivation claims about benchmark performance should cite an actual reproducible evaluation: code revision, real artifact/model identity, validated dataset, sample/seed, units, raw outputs, and methodology. If any critical link is missing, label the claim **exploratory** or **unverified** rather than presenting it as a proven result.
