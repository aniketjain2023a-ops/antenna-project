"""Optimizer A: SciPy Differential Evolution (Storn & Price, 1997).

DE is population-based and derivative-free, which suits tree-ensemble
surrogates whose response surface is piecewise constant (gradients are zero
almost everywhere). The objective is evaluated in vectorised form so a whole
population is predicted in one model call. Local polishing (L-BFGS-B) is
disabled because it relies on gradients.

Evaluation budget: total evaluations ~= popsize * n_params * (maxiter + 1);
``maxiter`` is derived from the requested budget.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import differential_evolution

from optimization.base import BestTracker, OptimizationOutcome, finalise
from optimization.constraints import bounds_array
from optimization.objectives import ObjectiveConfig, SurrogateEvaluator, make_scalar_objective
from utils.compat import seed_kwargs


def run_differential_evolution(
    evaluator: SurrogateEvaluator,
    cfg: ObjectiveConfig,
    bounds: dict[str, tuple[float, float]],
    seed: int = 42,
    budget: int = 1500,
    popsize: int = 15,
    mutation: tuple[float, float] = (0.5, 1.0),
    recombination: float = 0.7,
    tol: float = 1e-6,
) -> OptimizationOutcome:
    bnds = bounds_array(bounds, evaluator.features)
    d = len(evaluator.features)
    pop = popsize * d
    maxiter = max(1, budget // pop - 1)
    tracker = BestTracker(make_scalar_objective(evaluator, cfg))

    def vec_obj(x):  # SciPy passes shape (n_params, S) when vectorized=True
        return tracker(np.asarray(x).T)

    evaluator.n_evals = 0
    t0 = time.perf_counter()
    res = differential_evolution(
        vec_obj, bounds=bnds.tolist(), maxiter=maxiter, popsize=popsize, mutation=mutation,
        recombination=recombination, tol=tol, polish=False, init="latinhypercube",
        updating="deferred", vectorized=True, **seed_kwargs(differential_evolution, seed),
    )
    settings = {"popsize_multiplier": popsize, "population": pop, "maxiter": maxiter,
                "mutation": list(mutation), "recombination": recombination, "tol": tol,
                "polish": False, "init": "latinhypercube", "strategy": "best1bin"}
    return finalise("differential_evolution", settings, res.x, bnds, evaluator, cfg, tracker, budget, t0,
                    bool(res.success) or res.nit >= maxiter, str(res.message))
