"""Candidate trade-off analysis, ranking and plain-language explanations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import qmc

from optimization.constraints import bounds_array
from optimization.objectives import DIRECTION, ObjectiveConfig, SurrogateEvaluator, evaluate_objective
from utils.compat import seed_kwargs


def pareto_mask(values: np.ndarray, minimize: list[bool]) -> np.ndarray:
    """Boolean mask of non-dominated rows. ``values`` has shape (n, m)."""
    v = np.where(np.array(minimize)[None, :], values, -values)  # convert all to minimisation
    n = len(v)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominated_by = np.all(v <= v[i], axis=1) & np.any(v < v[i], axis=1)
        if dominated_by.any():
            mask[i] = False
    return mask


def sample_design_space(evaluator: SurrogateEvaluator, cfg: ObjectiveConfig, bounds: dict,
                        n: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Space-filling sample of predicted designs with feasibility and Pareto flags."""
    bnds = bounds_array(bounds, evaluator.features)
    sampler = qmc.LatinHypercube(d=len(evaluator.features), **seed_kwargs(qmc.LatinHypercube, seed))
    X = qmc.scale(sampler.random(n), bnds[:, 0], bnds[:, 1])
    preds = evaluator(X)
    ev = evaluate_objective(cfg, preds)
    df = pd.DataFrame(X, columns=evaluator.features)
    for t, (m, s) in preds.items():
        df[t] = m
        df[f"{t}_std"] = s if s is not None else np.nan
    df["objective"] = ev["penalised"]
    df["feasible"] = ev["violation"] <= 1e-12
    targets = [t for t in preds if t in DIRECTION]
    if len(targets) >= 2:
        df["pareto"] = pareto_mask(df[targets].to_numpy(), [DIRECTION[t] == "minimize" for t in targets])
    else:
        df["pareto"] = False
    return df


def rank_candidates(candidates: list[dict], cfg: ObjectiveConfig) -> pd.DataFrame:
    """Rank candidate designs: feasible first, then by penalised objective (lower = better)."""
    rows = []
    for c in candidates:
        row = {"source": c.get("source", "candidate"), **c["parameters"]}
        for t, v in c["predictions"].items():
            row[t] = v
            u = c["uncertainty"].get(t)
            row[f"{t}_std"] = u
        row["feasible"] = c["feasible"]
        row["objective"] = c["objective_penalised"]
        row["total_violation"] = sum(c["violations"].values())
        row["explanation"] = explain_candidate(c, cfg)
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.sort_values(["feasible", "objective"], ascending=[False, True]).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def explain_candidate(c: dict, cfg: ObjectiveConfig) -> str:
    parts = []
    if c["feasible"]:
        margins = []
        for con in cfg.constraints:
            v = c["predictions"].get(con.target)
            if v is not None:
                margin = (con.limit - v) if con.operator == "<=" else (v - con.limit)
                margins.append(f"{con.target} margin {margin:+.2f}")
        parts.append("predicted feasible" + (f" ({', '.join(margins)})" if margins else ""))
    else:
        viol = ", ".join(f"{k} violated by {v:.2f}" for k, v in c["violations"].items() if v > 0)
        parts.append(f"INFEASIBLE: {viol}")
    if cfg.kind == "max_gain":
        parts.append(f"ranked by gain {c['predictions'].get(cfg.gain_target, float('nan')):.2f} dBi")
    elif cfg.kind == "min_s11":
        parts.append(f"ranked by S11 {c['predictions'].get(cfg.s11_target, float('nan')):.2f} dB")
    elif cfg.kind in ("minimize_target", "maximize_target"):
        parts.append(f"ranked by {cfg.target} {c['predictions'].get(cfg.target, float('nan')):.3f}")
    else:
        parts.append(f"normalized trade-off score {-c['objective_raw']:.3f} (higher is better)")
    unc = [f"{t} +/-{u:.2f}" for t, u in c["uncertainty"].items() if u is not None]
    if unc:
        parts.append("approx. uncertainty " + ", ".join(unc))
    return "; ".join(parts)
