"""
Evolutionary Search for Neural Architectures
==============================================

Population-based evolutionary search using tropical expressivity as
the fitness function.

Algorithm
---------
1. Initialize population of random architectures.
2. Evaluate fitness (tropical expressivity score) for each.
3. Repeat for N generations:
   a. Tournament selection of parents.
   b. Crossover + mutation to produce offspring.
   c. Evaluate offspring fitness.
   d. Elitist replacement: keep best from parents + offspring.
4. Return the best architecture found.
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from canopy.nas.architecture import ArchitectureSpec, SearchSpace
from canopy.nas.tropical_score import tropical_expressivity_score


# ---------------------------------------------------------------------------
#  Search result
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Result of an evolutionary search run."""

    best_arch: ArchitectureSpec
    best_score: int
    history: List[Dict]  # per-generation stats
    total_evaluations: int
    wall_time_seconds: float
    population_final: List[Tuple[ArchitectureSpec, int]]

    def __repr__(self) -> str:
        return (
            f"SearchResult(best_score={self.best_score}, "
            f"generations={len(self.history)}, "
            f"evals={self.total_evaluations}, "
            f"time={self.wall_time_seconds:.1f}s)"
        )


# ---------------------------------------------------------------------------
#  Evolutionary Search
# ---------------------------------------------------------------------------

class EvolutionarySearch:
    """
    Evolutionary NAS with tropical expressivity fitness.

    Parameters
    ----------
    search_space : SearchSpace
        Defines random sampling, mutation, and crossover.
    population_size : int
        Number of architectures in the population.
    n_generations : int
        Number of evolutionary generations.
    tournament_size : int
        Number of candidates in tournament selection.
    mutation_rate : float
        Probability of mutation (vs. crossover) per offspring.
    elite_ratio : float
        Fraction of population carried over as elites.
    scoring_mode : str
        Passed to tropical_expressivity_score.
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        population_size: int = 50,
        n_generations: int = 100,
        tournament_size: int = 5,
        mutation_rate: float = 0.7,
        elite_ratio: float = 0.1,
        scoring_mode: str = "generic",
        seed: Optional[int] = None,
        fitness_fn: Optional[Callable[[ArchitectureSpec], float]] = None,
    ):
        self.search_space = search_space
        self.population_size = population_size
        self.n_generations = n_generations
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
        self.scoring_mode = scoring_mode
        self.seed = seed
        self.fitness_fn = fitness_fn

    def search(
        self,
        *,
        callback: Optional[Callable[[int, Dict], None]] = None,
        max_evaluations: Optional[int] = None,
    ) -> SearchResult:
        """
        Run evolutionary search with a strict limit on *unique* fitness calls.

        Duplicate architectures are retrieved from the cache without
        consuming evaluation budget. If a budget is exhausted during
        population initialization or a generation, return the best
        evaluated candidates without scoring any unevaluated offspring.
        """
        if self.population_size < 1:
            raise ValueError("population_size must be positive")
        if self.tournament_size < 1:
            raise ValueError("tournament_size must be positive")
        if max_evaluations is not None and (
            isinstance(max_evaluations, bool)
            or not isinstance(max_evaluations, int)
            or max_evaluations < 1
        ):
            raise ValueError("max_evaluations must be a positive integer or None")

        rng = np.random.default_rng(self.seed)
        t_start = time.perf_counter()

        cache: Dict[ArchitectureSpec, float] = {}
        total_evals = 0

        class EvaluationBudgetReached(Exception):
            """Internal signal that no new unique score may be computed."""

        def evaluate(arch: ArchitectureSpec) -> float:
            nonlocal total_evals
            if arch not in cache:
                if max_evaluations is not None and total_evals >= max_evaluations:
                    raise EvaluationBudgetReached
                if self.fitness_fn is not None:
                    cache[arch] = self.fitness_fn(arch)
                else:
                    cache[arch] = float(
                        tropical_expressivity_score(arch, mode=self.scoring_mode)
                    )
                total_evals += 1
            return cache[arch]

        # Evaluate initial candidates one at a time. The budget can be
        # smaller than population_size, so initialization may be partial.
        population: List[ArchitectureSpec] = []
        scores: List[float] = []
        for _ in range(self.population_size):
            arch = self.search_space.random(rng)
            try:
                score = evaluate(arch)
            except EvaluationBudgetReached:
                break
            population.append(arch)
            scores.append(score)
            if max_evaluations is not None and total_evals >= max_evaluations:
                break

        history: List[Dict] = []
        n_elite = max(1, int(self.elite_ratio * len(population)))

        for gen in range(self.n_generations):
            if max_evaluations is not None and total_evals >= max_evaluations:
                break

            ranked = sorted(
                zip(population, scores), key=lambda x: x[1], reverse=True
            )
            elites = [a for a, _ in ranked[:n_elite]]
            elite_scores = [s for _, s in ranked[:n_elite]]

            offspring: List[ArchitectureSpec] = []
            offspring_scores: List[float] = []
            n_offspring = len(population) - n_elite

            for _ in range(n_offspring):
                if rng.random() < self.mutation_rate:
                    parent = self._tournament_select(population, scores, rng)
                    child = self.search_space.mutate(parent, rng)
                else:
                    p1 = self._tournament_select(population, scores, rng)
                    p2 = self._tournament_select(population, scores, rng)
                    child = self.search_space.crossover(p1, p2, rng)

                try:
                    child_score = evaluate(child)
                except EvaluationBudgetReached:
                    break
                offspring.append(child)
                offspring_scores.append(child_score)

                if max_evaluations is not None and total_evals >= max_evaluations:
                    break

            # A partial generation keeps the un-replaced parents; otherwise
            # we would drop good candidates and shrink the population.
            unfilled = n_offspring - len(offspring)
            retained = ranked[n_elite : n_elite + unfilled]
            population = elites + offspring + [a for a, _ in retained]
            scores = elite_scores + offspring_scores + [s for _, s in retained]

            stats = {
                "generation": gen,
                "best_score": max(scores),
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "unique_archs": len(set(population)),
                "total_evals": total_evals,
                "cache_size": len(cache),
            }
            history.append(stats)
            if callback:
                callback(gen, stats)

            if max_evaluations is not None and total_evals >= max_evaluations:
                break

        ranked = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
        best_arch, best_score = ranked[0]
        return SearchResult(
            best_arch=best_arch,
            best_score=best_score,
            history=history,
            total_evaluations=total_evals,
            wall_time_seconds=time.perf_counter() - t_start,
            population_final=ranked,
        )

    def _tournament_select(
        self,
        population: List[ArchitectureSpec],
        scores: List[int],
        rng: np.random.Generator,
    ) -> ArchitectureSpec:
        """Select the best of k random candidates."""
        indices = rng.choice(
            len(population), size=min(self.tournament_size, len(population)),
            replace=False,
        )
        best_idx = max(indices, key=lambda i: scores[i])
        return population[best_idx]
