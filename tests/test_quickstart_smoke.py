"""Exercise the documented beginner commands through the actual module entry point."""

import json
import subprocess
import sys


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "canopy.cli", *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )


def test_beginner_quickstart_runs_without_benchmark_data():
    version = _run_cli("--version")
    assert "CANOPY" in version.stdout

    score = _run_cli("score", "--widths", "2,4,1")
    assert "architecture-level region-count upper bound" in score.stdout
    assert score.stderr == ""


def test_advanced_quickstart_json_is_machine_readable():
    result = _run_cli("score", "--widths", "2,64,64,1", "--mode", "tight", "--json")
    payload = json.loads(result.stdout)
    assert payload["layer_widths"] == [2, 64, 64, 1]
    assert payload["mode"] == "tight"
    assert payload["score_units"] == "count_bound"
    assert isinstance(payload["score"], str) and payload["score"].isdecimal()
    assert result.stderr == ""
