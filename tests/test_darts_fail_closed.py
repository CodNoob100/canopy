"""Offline fail-closed regressions for optional NAS-Bench-301 integration."""

import sys
import types

import pytest


def test_darts_evaluation_refuses_missing_surrogate(monkeypatch):
    from examples import run_darts_evaluation as darts

    stub = types.ModuleType("nasbench301")
    stub.load_ensemble = lambda path: (_ for _ in ()).throw(
        FileNotFoundError("surrogate models not installed")
    )
    monkeypatch.setitem(sys.modules, "nasbench301", stub)
    with pytest.raises(RuntimeError, match="surrogate models are required"):
        darts.run_evaluation(num_samples=0)


def test_darts_evaluation_refuses_failed_surrogate_query(monkeypatch):
    from examples import run_darts_evaluation as darts

    stub = types.ModuleType("nasbench301")
    stub.__path__ = []
    stub.load_ensemble = lambda path: types.SimpleNamespace(
        convert_representation=lambda genotype, fmt: {},
        model=types.SimpleNamespace(
            query=lambda config: (_ for _ in ()).throw(
                ValueError("upstream model failure")
            )
        ),
    )
    representations = types.ModuleType("nasbench301.representations")
    representations.Genotype = lambda **kwargs: kwargs
    api = types.ModuleType("nasbench301.api")
    api.fixed_hyperparameters = {}
    monkeypatch.setitem(sys.modules, "nasbench301", stub)
    monkeypatch.setitem(sys.modules, "nasbench301.representations", representations)
    monkeypatch.setitem(sys.modules, "nasbench301.api", api)
    monkeypatch.setattr(darts, "tropical_expressivity_score_darts", lambda arch: 0.0)

    with pytest.raises(RuntimeError, match="surrogate query failed for sample 0"):
        darts.run_evaluation(num_samples=1)
