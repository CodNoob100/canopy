"""Tests for the training-free scoring CLI."""

import json

import pytest

from canopy.cli import main, parse_widths


def test_generic_score_is_count_bound(capsys):
    assert main(["score", "--widths", "2,4,1", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["score"] == "11"
    assert result["score_units"] == "count_bound"
    assert result["mode"] == "generic"
    assert result["layer_widths"] == [2, 4, 1]
    assert "not an exact" in result["score_interpretation"]


def test_tight_score_is_not_greater_than_generic(capsys):
    main(["score", "--widths", "2,4,4,1", "--mode", "generic", "--json"])
    generic = int(json.loads(capsys.readouterr().out)["score"])

    main(["score", "--widths", "2,4,4,1", "--mode", "tight", "--json"])
    tight = int(json.loads(capsys.readouterr().out)["score"])

    assert 0 < tight <= generic


def test_default_plain_output(capsys):
    assert main(["score", "--widths", "2,4,1"]) == 0
    assert "11 (architecture-level region-count upper bound, generic)" in capsys.readouterr().out


@pytest.mark.parametrize("bad", ["", "2", "2,0,1", "2,-1,1", "2,wat,1", "2,,1"])
def test_invalid_widths_are_rejected(bad):
    with pytest.raises(Exception) as exc:
        parse_widths(bad)
    assert "positive" in str(exc.value) or "at least" in str(exc.value)


def test_cli_requires_widths(capsys):
    with pytest.raises(SystemExit) as exit_result:
        main(["score"])
    assert exit_result.value.code == 2
    assert "--widths" in capsys.readouterr().err


def test_cli_rejects_bad_scoring_mode():
    with pytest.raises(SystemExit) as exit_result:
        main(["score", "--widths", "2,4,1", "--mode", "unknown"])
    assert exit_result.value.code == 2


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exit_result:
        main(["--version"])
    assert exit_result.value.code == 0
    assert "CANOPY" in capsys.readouterr().out
