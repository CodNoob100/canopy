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
        Run the evolutionary search.

        Parameters
        ----------
        callback : callable, optional
            Called after each generation with (generation, stats_dict).
        max_evaluations : int, optional
            Stop early if total evaluations exceed this.

        Returns
        -------
        SearchResult
        """
        rng = np.random.default_rng(self.seed)
        t_start = time.perf_counter()

        # Cache: arch → score
        cache: Dict[ArchitectureSpec, float] = {}
        total_evals = 0

        def evaluate(arch: ArchitectureSpec) -> float:
            nonlocal total_evals
            if arch not in cache:
                if self.fitness_fn is not None:
                    cache[arch] = self.fitness_fn(arch)
                else:
                    cache[arch] = float(tropical_expressivity_score(
                        arch, mode=self.scoring_mode
                    ))
                total_evals += 1
            return cache[arch]

        # Initialize population
        population: List[ArchitectureSpec] = [
            self.search_space.random(rng)
            for _ in range(self.population_size)
        ]
        scores = [evaluate(a) for a in population]

        history: List[Dict] = []
        n_elite = max(1, int(self.elite_ratio * self.population_size))

        for gen in range(self.n_generations):
            if max_evaluations and total_evals >= max_evaluations:
                break

            # Sort by score (descending — higher is more expressive)
            ranked = sorted(
                zip(population, scores), key=lambda x: x[1], reverse=True
            )

            # Elitism: keep top n_elite
            elites = [a for a, _ in ranked[:n_elite]]
            elite_scores = [s for _, s in ranked[:n_elite]]

            # Generate offspring
            offspring: List[ArchitectureSpec] = []
            n_offspring = self.population_size - n_elite

            for _ in range(n_offspring):
                if rng.random() < self.mutation_rate:
                    # Mutation
                    parent = self._tournament_select(
                        population, scores, rng
                    )
                    child = self.search_space.mutate(parent, rng)
                else:
                    # Crossover
                    p1 = self._tournament_select(population, scores, rng)
                    p2 = self._tournament_select(population, scores, rng)
                    child = self.search_space.crossover(p1, p2, rng)
                offspring.append(child)

            offspring_scores = [evaluate(a) for a in offspring]

            # New population = elites + offspring
            population = elites + offspring
            scores = elite_scores + offspring_scores

            # Stats
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

        # Final ranking
        ranked = sorted(
            zip(population, scores), key=lambda x: x[1], reverse=True
        )
        best_arch, best_score = ranked[0]
        wall_time = time.perf_counter() - t_start

        return SearchResult(
            best_arch=best_arch,
            best_score=best_score,
            history=history,
            total_evaluations=total_evals,
            wall_time_seconds=wall_time,
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
            len(population), size=self.tournament_size, replace=False
        )
        best_idx = max(indices, key=lambda i: scores[i])
        return population[best_idx]
