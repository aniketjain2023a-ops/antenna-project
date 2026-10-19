"""Objective functions evaluated on the trained surrogate models.

Supported objectives (all are MINIMISED internally)
---------------------------------------------------
``max_gain``  : maximise predicted gain subject to S11 <= limit (and any extra
                constraints).          raw = -Gain_dBi
``min_s11``   : minimise predicted S11 subject to Gain >= limit.
                                        raw = S11_dB
``tradeoff``  : weighted sum of NORMALISED desirabilities. Each target is first
                mapped to [0, 1] using its observed training range, with
                "better" = 1, so dB and dBi values are never added directly::

                    d_gain = (gain - gain_min) / (gain_max - gain_min)
                    d_s11  = (s11_max - s11) / (s11_max - s11_min)
                    raw    = -(w_g d_gain + w_s d_s11 [+ w_b d_bw]) / sum(w)

                Bandwidth participates only if a real bandwidth model exists.

Constraint handling: static penalty. Each violation is divided by its target's
training range (dimensionless) and multiplied by ``penalty_weight``::

    penalised = raw + penalty_weight * sum_i violation_i / range_i

Optional risk-aware mode (``kappa`` > 0) uses pessimistic predictions
``mean -/+ kappa * sigma`` where uncertainty is available, so candidates in
poorly-sampled regions are discouraged. This is a heuristic, not a guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import config
from ml.prediction import LoadedModel, check_feature_consistency, predict_batch
from optimization.constraints import Constraint

OBJECTIVE_DESCRIPTIONS = {
    "max_gain": "Maximize predicted gain subject to S11 <= limit",
    "min_s11": "Minimize predicted S11 subject to gain >= limit",
    "tradeoff": "Normalized weighted gain / S11 trade-off (optional constraints)",
}

DIRECTION = {"S11_dB": "minimize", "Gain_dBi": "maximize", "Bandwidth_MHz": "maximize"}


class SurrogateEvaluator:
    """Calls the trained models; counts every design evaluated."""

    def __init__(self, models: dict[str, LoadedModel]):
        if not models:
            raise ValueError("At least one trained model is required.")
        self.models = models
        self.features = check_feature_consistency(models)
        self.n_evals = 0

    def __call__(self, X) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        self.n_evals += X.shape[0]
        return predict_batch(self.models, X)


@dataclass
class ObjectiveConfig:
    kind: str
    constraints: list[Constraint] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    normalization: dict[str, tuple[float, float]] = field(default_factory=dict)
    kappa: float = 0.0
    penalty_weight: float = config.CONSTRAINT_PENALTY_WEIGHT
    s11_target: str = "S11_dB"
    gain_target: str = "Gain_dBi"

    def validate(self, available_targets: list[str]) -> None:
        if self.kind not in OBJECTIVE_DESCRIPTIONS:
            raise ValueError(f"Unknown objective '{self.kind}'.")
        if self.kind == "max_gain" and self.gain_target not in available_targets:
            raise ValueError(f"Objective requires a trained model for {self.gain_target}.")
        if self.kind == "min_s11" and self.s11_target not in available_targets:
            raise ValueError(f"Objective requires a trained model for {self.s11_target}.")
        if self.kind == "tradeoff":
            usable = {t: w for t, w in self.weights.items() if t in available_targets and w > 0}
            if not usable:
                raise ValueError("Trade-off objective needs at least one positive weight on a trained target.")
            missing_norm = [t for t in usable if t not in self.normalization]
            if missing_norm:
                raise ValueError(f"Missing normalization ranges for {missing_norm}.")
            unknown_dir = [t for t in usable if t not in DIRECTION]
            if unknown_dir:
                raise ValueError(f"No optimisation direction known for {unknown_dir}.")
        for c in self.constraints:
            if c.target not in available_targets:
                raise ValueError(f"Constraint on '{c.target}' but no trained model for it.")
            if c.target not in self.normalization:
                raise ValueError(f"Missing normalization range for constrained target '{c.target}'.")
        if self.kappa < 0:
            raise ValueError("kappa must be >= 0.")

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "description": OBJECTIVE_DESCRIPTIONS[self.kind],
            "weights": self.weights, "normalization": {k: list(v) for k, v in self.normalization.items()},
            "kappa": self.kappa, "penalty_weight": self.penalty_weight,
            "constraints": [c.to_dict() for c in self.constraints],
        }


def _pessimistic(target: str, mean: np.ndarray, std: np.ndarray | None, kappa: float) -> np.ndarray:
    if kappa <= 0 or std is None:
        return mean
    return mean + kappa * std if DIRECTION.get(target) == "minimize" else mean - kappa * std


def _desirability(target: str, values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    span = hi - lo if hi > lo else 1.0
    if DIRECTION[target] == "maximize":
        return (values - lo) / span
    return (hi - values) / span


def evaluate_objective(cfg: ObjectiveConfig, preds: dict) -> dict[str, np.ndarray]:
    """Return raw objective, normalized total violation and penalised objective arrays."""
    eff = {t: _pessimistic(t, m, s, cfg.kappa) for t, (m, s) in preds.items()}
    if cfg.kind == "max_gain":
        raw = -eff[cfg.gain_target]
    elif cfg.kind == "min_s11":
        raw = eff[cfg.s11_target].copy()
    else:
        usable = {t: w for t, w in cfg.weights.items() if t in eff and w > 0}
        total_w = sum(usable.values())
        score = sum(w * _desirability(t, eff[t], *cfg.normalization[t]) for t, w in usable.items())
        raw = -score / total_w
    n = len(raw)
    norm_violation = np.zeros(n)
    per_constraint = {}
    for c in cfg.constraints:
        lo, hi = cfg.normalization[c.target]
        v = c.violation(eff[c.target])
        per_constraint[c.describe()] = v
        norm_violation += v / (hi - lo if hi > lo else 1.0)
    return {
        "raw": np.asarray(raw, dtype=float),
        "violation": norm_violation,
        "per_constraint": per_constraint,
        "penalised": np.asarray(raw, dtype=float) + cfg.penalty_weight * norm_violation,
    }


def make_scalar_objective(evaluator: SurrogateEvaluator, cfg: ObjectiveConfig):
    """Return ``f(X) -> penalised objective`` accepting (n_features,) or (n, n_features)."""

    def f(X):
        X = np.asarray(X, dtype=float)
        single = X.ndim == 1
        res = evaluate_objective(cfg, evaluator(X))["penalised"]
        return float(res[0]) if single else res

    return f


def describe_point(evaluator: SurrogateEvaluator, cfg: ObjectiveConfig, x) -> dict:
    """Full breakdown (predictions, uncertainty, objective, violations) of one design."""
    preds = predict_batch(evaluator.models, np.atleast_2d(np.asarray(x, dtype=float)))
    ev = evaluate_objective(cfg, preds)
    return {
        "parameters": {f: float(v) for f, v in zip(evaluator.features, np.asarray(x, dtype=float).ravel())},
        "predictions": {t: float(m[0]) for t, (m, s) in preds.items()},
        "uncertainty": {t: (None if s is None else float(s[0])) for t, (m, s) in preds.items()},
        "objective_raw": float(ev["raw"][0]),
        "objective_penalised": float(ev["penalised"][0]),
        "violations": {k: float(v[0]) for k, v in ev["per_constraint"].items()},
        "feasible": bool(ev["violation"][0] <= 1e-12),
    }
