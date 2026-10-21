"""Regression tests for the multi-stage NAS pipeline."""

import pytest

from canopy.nas.pipeline import MultiStagePipeline


def test_empty_pool_returns_no_finalists_without_calling_stages():
    def should_not_run(_):
        raise AssertionError("Stage functions must not run for an empty pool")

    pipeline = MultiStagePipeline(
        score_fn=should_not_run,
        proxy_train_fn=should_not_run,
        full_train_fn=should_not_run,
    )

    assert pipeline.run([]) == []


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.1, float("nan")])
def test_invalid_filter_ratio_is_rejected(ratio):
    with pytest.raises(ValueError, match="filter_keep_ratio"):
        MultiStagePipeline(lambda _: 0, lambda _: 0, lambda _: 0, filter_keep_ratio=ratio)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_finalist_count_is_rejected(top_k):
    with pytest.raises(ValueError, match="proxy_keep_top_k"):
        MultiStagePipeline(lambda _: 0, lambda _: 0, lambda _: 0, proxy_keep_top_k=top_k)


def test_stage_order_and_finalist_ranking():
    scored, proxied, trained = [], [], []

    def score(arch):
        scored.append(arch)
        return int(arch)

    def proxy(arch):
        proxied.append(arch)
        return 10 - int(arch)

    def train(arch):
        trained.append(arch)
        return 2 * int(arch)

    pipeline = MultiStagePipeline(
        score_fn=score,
        proxy_train_fn=proxy,
        full_train_fn=train,
        filter_keep_ratio=0.5,
        proxy_keep_top_k=2,
    )
    finalists = pipeline.run(["0", "1", "2", "3", "4", "5"])

    assert scored == ["0", "1", "2", "3", "4", "5"]
    assert proxied == ["5", "4", "3"]
    assert trained == ["3", "4"]
    assert [candidate.arch_str for candidate in finalists] == ["4", "3"]
    assert [candidate.final_accuracy for candidate in finalists] == [8, 6]
