"""Regressions for strictly capped NAS evaluation budgets."""

import pytest

from canopy.nas.architecture import ArchitectureSpec, SearchSpace
from canopy.nas.search import EvolutionarySearch


class UniqueSearchSpace(SearchSpace):
    """Each operation produces a new, hashable architecture."""

    def __init__(self):
        self.counter = 0

    def _new(self):
        self.counter += 1
        return ArchitectureSpec(layer_widths=(1, self.counter, 1))

    def random(self, rng):
        return self._new()

    def mutate(self, arch, rng):
        return self._new()

    def crossover(self, parent_a, parent_b, rng):
        return self._new()


class ConstantSearchSpace(UniqueSearchSpace):
    def _new(self):
        return ArchitectureSpec(layer_widths=(1, 2, 1))


@pytest.mark.parametrize("budget", [1, 2, 4, 5, 6, 8, 10])
def test_strict_evaluation_budget_across_initial_and_partial_generations(budget):
    calls = []

    def fitness(arch):
        calls.append(arch)
        return float(arch.hidden_widths[0])

    search = EvolutionarySearch(
        search_space=UniqueSearchSpace(),
        population_size=5,
        n_generations=10,
        tournament_size=3,
        fitness_fn=fitness,
        seed=5,
    )
    result = search.search(max_evaluations=budget)

    assert result.total_evaluations == budget
    assert len(calls) == budget
    assert result.best_score == max(float(a.hidden_widths[0]) for a in calls)
    assert result.population_final
    if budget < 5:
        assert len(result.population_final) == budget
        assert result.history == []
    else:
        assert len(result.population_final) == 5
    assert all(step["total_evals"] <= budget for step in result.history)


def test_cached_architectures_do_not_consume_budget():
    calls = []
    search = EvolutionarySearch(
        search_space=ConstantSearchSpace(),
        population_size=8,
        n_generations=3,
        fitness_fn=lambda arch: calls.append(arch) or 3.0,
        seed=17,
    )

    result = search.search(max_evaluations=2)
    assert len(calls) == result.total_evaluations == 1
    assert len(result.history) == 3
    assert len(result.population_final) == 8


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_budget_is_rejected(budget):
    search = EvolutionarySearch(search_space=UniqueSearchSpace())
    with pytest.raises(ValueError, match="max_evaluations"):
        search.search(max_evaluations=budget)
