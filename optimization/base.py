"""Shared result type and post-processing for all optimizers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from optimization.constraints import enforce_bounds, within_bounds
from optimization.objectives import ObjectiveConfig, SurrogateEvaluator, describe_point

CANDIDATE_LABEL = "ML-selected candidate (surrogate prediction - NOT physically validated)"


@dataclass
class OptimizationOutcome:
    algorithm: str
    settings: dict
    candidate: dict                 # describe_point() output for the best design
    n_evals: int
    budget: int
    runtime_s: float
    history: list[float] = field(default_factory=list)   # best penalised objective so far
    success: bool = True
    message: str = ""
    within_bounds: bool = True
    bounds_clipped: bool = False
    label: str = CANDIDATE_LABEL


class BestTracker:
    """Wraps the vectorised objective to record the best-so-far trace."""

    def __init__(self, f):
        self.f = f
        self.best = np.inf
        self.history: list[float] = []

    def __call__(self, X):
        vals = self.f(X)
        arr = np.atleast_1d(vals)
        for v in arr:
            self.best = min(self.best, float(v))
            self.history.append(self.best)
        return vals


def finalise(algorithm: str, settings: dict, x_best, bnds: np.ndarray, evaluator: SurrogateEvaluator,
             cfg: ObjectiveConfig, tracker: BestTracker, budget: int, t0: float,
             success: bool, message: str) -> OptimizationOutcome:
    x_best = np.asarray(x_best, dtype=float)
    inside = within_bounds(x_best, bnds)
    clipped = False
    if not inside:  # should not happen: bounded optimizers; kept as a hard safety net
        x_best = enforce_bounds(x_best, bnds)
        clipped = True
    n_evals_search = evaluator.n_evals
    candidate = describe_point(evaluator, cfg, x_best)
    return OptimizationOutcome(
        algorithm=algorithm, settings=settings, candidate=candidate, n_evals=n_evals_search, budget=budget,
        runtime_s=time.perf_counter() - t0, history=tracker.history, success=success, message=message,
        within_bounds=within_bounds(x_best, bnds), bounds_clipped=clipped,
    )
