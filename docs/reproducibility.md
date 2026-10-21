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

## Missing benchmark entries

When a proposed architecture is absent from the benchmark, its accuracy is
**unknown**, not 0%. The evaluator now writes JSON `null` for an unavailable
final accuracy or a trajectory with no observed accuracies so far. Once an
available architecture is encountered, the trajectory retains the highest
*observed* accuracy. Results carry `missing_benchmark_lookups` and
`trajectory_lookup_available`; CSV rows include `benchmark_lookup_available`.
A genuine benchmark accuracy of 0.0 remains numeric zero.

A synthetic benchmark that represents only a subset of the NAS search space may
produce many missing lookups. The labels and counts make this visible instead
of disguising gaps as 0%-accuracy observations. This is a compatibility change
for downstream analyses that previously expected every accuracy to be numeric:
handle `null`/empty cells explicitly when plotting or computing statistics.
