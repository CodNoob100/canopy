"""Regression coverage for evaluation provenance and deterministic baselines."""

import json

import numpy as np
import pytest

from canopy.evaluation.runner import ExperimentRunner
from canopy.nas.architecture import NASBench201SearchSpace
from canopy.nas.benchmark import SyntheticBenchmark


def make_runner(tmp_path, **kwargs):
    benchmark = SyntheticBenchmark(n_architectures=100, seed=42)
    return ExperimentRunner(
        search_space=NASBench201SearchSpace(),
        benchmark=benchmark,
        strategies=["random", "tropical_generic"],
        population_size=8,
        n_generations=4,
        max_evaluations=15,
        output_dir=str(tmp_path),
        **kwargs,
    )


def test_random_baseline_repeats_for_same_seed(tmp_path):
    runner = make_runner(tmp_path)
    a = runner.run_seed(23)
    b = runner.run_seed(23)

    for key in ("random", "tropical_generic"):
        assert a[key]["final_fitness_score"] == b[key]["final_fitness_score"]
        assert a[key]["total_evaluations"] == b[key]["total_evaluations"]
        assert a[key]["trajectory_accuracy"] == b[key]["trajectory_accuracy"]
        assert a[key]["trajectory_evals"] == b[key]["trajectory_evals"]


def test_random_baseline_does_not_touch_global_numpy_randomness(tmp_path):
    runner = make_runner(tmp_path)
    initial_state = np.random.get_state()
    try:
        generator = runner._get_fitness_fn("random", seed=19)
        for _ in range(7):
            generator(None)
        result_state = np.random.get_state()
        assert all(
            np.array_equal(before, after)
            if isinstance(before, np.ndarray) else before == after
            for before, after in zip(initial_state, result_state)
        )
    finally:
        np.random.set_state(initial_state)


def test_manifest_classifies_synthetic_data_and_score_units(tmp_path):
    runner = make_runner(tmp_path)
    runner.run_all(n_seeds=1)

    manifests = list(tmp_path.glob("experiment_manifest_*.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))

    assert manifest["manifest_schema_version"] == 1
    assert manifest["benchmark"]["kind"] == "synthetic"
    assert manifest["benchmark"]["accuracy_source"] == "synthetic_simulation"
    assert manifest["benchmark"]["source_sha256"] is None
    assert manifest["configuration"]["seeds"] == [0]
    assert manifest["configuration"]["max_evaluations"] == 15
    assert manifest["configuration"]["fitness"]["random"]["units"] == "unit_interval_random"
    assert manifest["configuration"]["fitness"]["tropical_generic"]["units"] == "count_bound"
    assert isinstance(manifest["package_versions"], dict)
    assert (tmp_path / manifest["files"]["results_json"]).exists()
    assert (tmp_path / manifest["files"]["trajectory_csv"]).exists()

    result_file = tmp_path / manifest["files"]["results_json"]
    results = json.loads(result_file.read_text(encoding="utf-8"))
    assert results[0]["results"]["random"]["score_units"] == "unit_interval_random"
    assert results[0]["results"]["random"]["total_evaluations"] <= 15


def test_unknown_benchmark_must_not_be_classified_real(tmp_path):
    runner = make_runner(tmp_path)
    runner.benchmark = object()
    data = runner._benchmark_metadata()
    assert data["kind"] == "unknown"
    assert data["accuracy_source"] == "unclassified_benchmark"


@pytest.mark.parametrize("strategies", [[], ["made_up"]])
def test_invalid_strategies_fail_early(tmp_path, strategies):
    with pytest.raises(ValueError, match="strategies"):
        ExperimentRunner(
            search_space=NASBench201SearchSpace(),
            benchmark=SyntheticBenchmark(n_architectures=10),
            strategies=strategies, output_dir=str(tmp_path),
        )


def test_checksum_is_explicitly_not_verified(tmp_path):
    runner = make_runner(tmp_path, benchmark_sha256="a" * 64)
    runner.benchmark = object()
    metadata = runner._benchmark_metadata()
    assert metadata["source_sha256"] == "a" * 64
    assert metadata["checksum_status"] == "user_supplied_not_verified"


def test_invalid_checksum_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="benchmark_sha256"):
        make_runner(tmp_path, benchmark_sha256="unverified")


def test_missing_benchmark_entries_are_null_not_zero(tmp_path):
    runner = make_runner(tmp_path)
    runner.benchmark = type(
        "NoCoverageBenchmark", (), {"query_by_arch": lambda self, arch: None}
    )()
    out = runner.run_seed(8)

    for name, metrics in out.items():
        assert metrics["final_true_accuracy"] is None
        assert metrics["missing_benchmark_lookups"] == metrics["total_evaluations"]
        assert all(value is None for value in metrics["trajectory_accuracy"])
        assert not any(metrics["trajectory_lookup_available"])


def test_true_zero_is_distinct_from_missing(tmp_path):
    runner = make_runner(tmp_path)
    runner.benchmark = type(
        "ZeroAccuracyBenchmark", (), {
            "query_by_arch": lambda self, arch: type("Entry", (), {"test_accuracy": 0.0})()
        }
    )()
    metrics = runner.run_seed(8)["random"]
    assert metrics["final_true_accuracy"] == 0.0
    assert all(value == 0.0 for value in metrics["trajectory_accuracy"])
    assert all(metrics["trajectory_lookup_available"])
    assert metrics["missing_benchmark_lookups"] == 0


def test_entry_with_missing_accuracy_remains_unknown(tmp_path):
    """An entry may exist without a usable accuracy measurement."""
    runner = make_runner(tmp_path)
    runner.benchmark = type(
        "MissingAccuracyBenchmark", (), {
            "query_by_arch": lambda self, arch: type("Entry", (), {"test_accuracy": None})()
        }
    )()
    result = runner.run_seed(8)["random"]
    assert result["final_true_accuracy"] is None
    assert result["missing_benchmark_lookups"] == result["total_evaluations"]
    assert all(value is None for value in result["trajectory_accuracy"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 100.1])
def test_invalid_numeric_accuracy_rejected(tmp_path, value):
    runner = make_runner(tmp_path)
    entry = type("Entry", (), {"test_accuracy": value})()
    with pytest.raises(ValueError, match="Benchmark test accuracy"):
        runner._observed_accuracy(entry)


def test_darts_example_fitness_override_forwards_seed(tmp_path):
    """The DARTS script must work with ExperimentRunner.run_seed(seed=...)."""
    from examples.run_darts_search_trajectory import configure_darts_fitness

    runner = make_runner(tmp_path)
    configure_darts_fitness(runner)

    first = runner._get_fitness_fn("random", seed=23)
    second = runner._get_fitness_fn("random", seed=23)
    other = runner._get_fitness_fn("random", seed=24)
    assert first(None) == second(None)
    assert first(None) != other(None)
    assert callable(runner._get_fitness_fn("tropical_generic", seed=23))
