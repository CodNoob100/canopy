"""Offline tests for the real-data-only evaluation CLI."""

import hashlib
import json

import pytest

from canopy import cli
from canopy.nas.architecture import NASBench201SearchSpace
from canopy.nas.benchmark import BenchmarkEntry


class FakeRealBenchmark:
    calls = []

    def __init__(self, data_path, dataset, hp):
        self.calls.append((data_path, dataset, hp))
        self.dataset = dataset
        self.search_space = NASBench201SearchSpace()

    @property
    def total_architectures(self):
        return 15625

    def query_by_index(self, i):
        arch = self.search_space.index_to_arch(i)
        return BenchmarkEntry(
            arch=arch,
            index=int(i),
            test_accuracy=float(60 + i % 29),
            valid_accuracy=50.0,
            params=float(arch.total_params),
            flops=float(arch.total_params * 2),
            dataset=self.dataset,
        )


def fake_backend(monkeypatch):
    import canopy.nas.benchmark as mod
    FakeRealBenchmark.calls = []
    monkeypatch.setattr(mod, "NATSBenchAPI", FakeRealBenchmark)


def test_real_dataset_writes_reproducible_report(tmp_path, monkeypatch):
    fake_backend(monkeypatch)
    data = tmp_path / "real.bin"
    data.write_bytes(b"offline fixture")
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    out = tmp_path / "results"
    command = [
        "evaluate", "--benchmark-data", str(data), "--dataset", "cifar100",
        "--seed", "42", "--limit", "20", "--output", str(out),
        "--sha256", digest,
    ]
    assert cli.main(command) == 0
    report = json.loads((out / "benchmark_comparison.json").read_text())
    assert FakeRealBenchmark.calls == [(str(data), "cifar100", "200")]
    assert report["benchmark"]["kind"] == "real_nats_bench"
    assert report["benchmark"]["checksum_status"] == "user_supplied_digest_matched_file"
    assert report["sample"]["seed"] == 42
    assert len(set(report["sample"]["indices"])) == 20
    assert len(report["metrics"]) == 5


def test_fixed_seed_reproducibility(tmp_path, monkeypatch):
    fake_backend(monkeypatch)
    data = tmp_path / "bench.bin"
    data.write_bytes(b"fixture")
    base = [
        "evaluate", "--benchmark-data", str(data), "--dataset", "cifar10",
        "--seed", "17", "--limit", "15",
    ]
    assert cli.main(base + ["--output", str(tmp_path / "a")]) == 0
    assert cli.main(base + ["--output", str(tmp_path / "b")]) == 0
    a = json.loads((tmp_path / "a" / "benchmark_comparison.json").read_text())
    b = json.loads((tmp_path / "b" / "benchmark_comparison.json").read_text())
    assert a == b
    assert a["benchmark"]["checksum_status"] == "not_verified"


def test_missing_dataset_fails_before_loading(tmp_path, monkeypatch):
    fake_backend(monkeypatch)
    with pytest.raises(SystemExit) as error:
        cli.main([
            "evaluate", "--benchmark-data", str(tmp_path / "missing"),
            "--dataset", "cifar10", "--seed", "2",
            "--output", str(tmp_path / "out"),
        ])
    assert error.value.code == 2
    assert FakeRealBenchmark.calls == []
    assert not (tmp_path / "out").exists()


def test_wrong_digest_refuses_loading(tmp_path, monkeypatch):
    fake_backend(monkeypatch)
    data = tmp_path / "bench"
    data.write_bytes(b"fixture")
    with pytest.raises(SystemExit):
        cli.main([
            "evaluate", "--benchmark-data", str(data),
            "--dataset", "cifar10", "--seed", "2",
            "--sha256", "0" * 64, "--output", str(tmp_path / "out"),
        ])
    assert FakeRealBenchmark.calls == []


def test_missing_lookup_fails_without_output(tmp_path, monkeypatch):
    import canopy.nas.benchmark as mod

    class Incomplete(FakeRealBenchmark):
        def query_by_index(self, i):
            return None

    monkeypatch.setattr(mod, "NATSBenchAPI", Incomplete)
    data = tmp_path / "bench"
    data.write_bytes(b"fixture")
    out = tmp_path / "out"
    with pytest.raises(SystemExit):
        cli.main([
            "evaluate", "--benchmark-data", str(data),
            "--dataset", "cifar10", "--seed", "2",
            "--limit", "10", "--output", str(out),
        ])
    assert not out.exists()


@pytest.mark.parametrize("limit", ["0", "1", "1001", "invalid"])
def test_bad_limit_rejected(tmp_path, limit):
    with pytest.raises(SystemExit) as error:
        cli.main([
            "evaluate", "--benchmark-data", "missing",
            "--dataset", "cifar100", "--seed", "42",
            "--limit", limit, "--output", str(tmp_path / "out"),
        ])
    assert error.value.code == 2


def test_existing_output_not_overwritten(tmp_path, monkeypatch):
    fake_backend(monkeypatch)
    data = tmp_path / "bench"
    data.write_bytes(b"fixture")
    out = tmp_path / "out"
    out.mkdir()
    (out / "important").write_text("keep")
    with pytest.raises(SystemExit):
        cli.main([
            "evaluate", "--benchmark-data", str(data),
            "--dataset", "cifar10", "--seed", "2", "--output", str(out),
        ])
    assert (out / "important").read_text() == "keep"


def test_undefined_statistics_not_reported_as_zero():
    assert cli._finite_or_none(float("nan")) is None
    assert cli._finite_or_none(float("inf")) is None
    assert cli._finite_or_none(0) == 0.0
