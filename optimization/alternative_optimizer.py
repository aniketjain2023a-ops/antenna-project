"""Optimizer B (and C): documented alternatives to Differential Evolution.

B. ``multistart_nelder_mead`` - Latin-hypercube multi-start Nelder-Mead
   (Nelder & Mead, 1965; McKay et al., 1979). A space-filling batch of
   starting points is screened with the surrogate; the best few seed bounded
   Nelder-Mead simplex searches (SciPy ``minimize(method="Nelder-Mead",
   bounds=...)``). Derivative-free and local, so the multi-start provides
   global coverage.

C. ``dual_annealing`` - SciPy generalized simulated annealing (Tsallis &
   Stariolo, 1996; Xiang et al., 2013). Gradient-based local search is
   disabled (``no_local_search=True``) because tree surrogates are piecewise
   constant.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import dual_annealing, minimize
from scipy.stats import qmc

from optimization.base import BestTracker, OptimizationOutcome, finalise
from optimization.constraints import bounds_array
from optimization.objectives import ObjectiveConfig, SurrogateEvaluator, make_scalar_objective
from utils.compat import seed_kwargs


def run_multistart_nelder_mead(
    evaluator: SurrogateEvaluator,
    cfg: ObjectiveConfig,
    bounds: dict[str, tuple[float, float]],
    seed: int = 42,
    budget: int = 1500,
    n_starts: int = 5,
    screen_fraction: float = 0.4,
) -> OptimizationOutcome:
    bnds = bounds_array(bounds, evaluator.features)
    d = len(evaluator.features)
    tracker = BestTracker(make_scalar_objective(evaluator, cfg))
    evaluator.n_evals = 0
    t0 = time.perf_counter()

    n_screen = max(n_starts, int(budget * screen_fraction))
    sampler = qmc.LatinHypercube(d=d, **seed_kwargs(qmc.LatinHypercube, seed))
    X0 = qmc.scale(sampler.random(n_screen), bnds[:, 0], bnds[:, 1])
    screen_vals = np.asarray(tracker(X0))
    starts = X0[np.argsort(screen_vals)[:n_starts]]

    remaining = max(n_starts * 10, budget - n_screen)
    per_start = max(10, remaining // n_starts)
    best_x, best_f = X0[int(np.argmin(screen_vals))], float(np.min(screen_vals))
    messages = []
    for x0 in starts:
        res = minimize(lambda x: tracker(x), x0, method="Nelder-Mead", bounds=bnds.tolist(),
                       options={"maxfev": per_start, "xatol": 1e-4, "fatol": 1e-6})
        messages.append(res.message)
        if res.fun < best_f:
            best_x, best_f = res.x, float(res.fun)
    settings = {"n_screen_lhs": n_screen, "n_starts": n_starts, "maxfev_per_start": per_start,
                "xatol": 1e-4, "fatol": 1e-6}
    return finalise("multistart_nelder_mead", settings, best_x, bnds, evaluator, cfg, tracker, budget, t0,
                    True, f"{n_starts} local searches completed")


def run_dual_annealing(
    evaluator: SurrogateEvaluator,
    cfg: ObjectiveConfig,
    bounds: dict[str, tuple[float, float]],
    seed: int = 42,
    budget: int = 1500,
) -> OptimizationOutcome:
    bnds = bounds_array(bounds, evaluator.features)
    tracker = BestTracker(make_scalar_objective(evaluator, cfg))
    evaluator.n_evals = 0
    t0 = time.perf_counter()
    res = dual_annealing(lambda x: tracker(x), bounds=bnds.tolist(), maxfun=budget, maxiter=budget,
                         no_local_search=True, **seed_kwargs(dual_annealing, seed))
    settings = {"maxfun": budget, "no_local_search": True, "visit": 2.62, "accept": -5.0}
    return finalise("dual_annealing", settings, res.x, bnds, evaluator, cfg, tracker, budget, t0,
                    bool(res.success), " ".join(res.message) if isinstance(res.message, list) else str(res.message))
