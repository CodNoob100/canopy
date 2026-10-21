# Reproducible NAS experiment manifests

`ExperimentRunner.run_all()` produces three files within the configured output
directory: `search_results_<run-id>.json`, `search_trajectory_<run-id>.csv`,
and `experiment_manifest_<run-id>.json`. The manifest is intended to make
evaluations auditable rather than to certify the accuracy of the benchmark.

The manifest records the Python and dependency versions, source revision (when
Git metadata is available), search space, evaluation budget, seed list,
benchmark kind, and scoring interpretations. It also records the generated
results filenames so reports can be associated with their provenance.

**Synthetic data are labeled `synthetic_simulation`, never real benchmark
ground truth.** The synthetic generator derives its simulated accuracies
partly from tropical scores and therefore cannot independently validate
a score-vs-accuracy correlation.

The random strategy uses an independent `numpy.random.default_rng(seed)`
per run. Running the same configuration with the same seed will reproduce
the random ranking trajectory, subject to stable dependencies and data.
Wall-clock time and generated run filenames are not deterministic.

For real NATS-Bench experiments, optional runner arguments
`benchmark_source_id="my-release-id"` and
`benchmark_sha256="<64-hex-digest>"` record a benchmark identifier and
caller-provided checksum; **the runner does not verify the checksum**.
Record an independently verified value if you need a trusted provenance
trail. Do not include sensitive local filesystem paths in the identifier.

The `score_units` field is `count_bound` for tropical generic/tight
bounds, `unit_interval_random` for the random baseline, and
`benchmark_parameter_score` for parameter count (which may have
benchmark-specific units). These are **fitness scores**, not accuracies;
the CSV `best_test_accuracy` column remains as a legacy column name and
must be interpreted with its `accuracy_source` column.

This PR adds metadata to the existing `ExperimentRunner`; a public
`canopy evaluate` CLI remains a separate feature requiring explicit
benchmark path, dataset, and seed validation.
