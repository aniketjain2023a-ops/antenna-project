"""Constraint definitions and bound handling for surrogate-based optimization.

Constraints are evaluated on surrogate PREDICTIONS. A candidate that satisfies
them is "predicted feasible", which is not the same as physically verified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from utils.validation import ValidationError, validate_bounds


@dataclass(frozen=True)
class Constraint:
    target: str
    operator: str     # "<=" or ">="
    limit: float

    def __post_init__(self):
        if self.operator not in ("<=", ">="):
            raise ValueError(f"Unsupported operator '{self.operator}'.")
        if not np.isfinite(self.limit):
            raise ValueError("Constraint limit must be finite.")

    def violation(self, values) -> np.ndarray:
        """Non-negative violation magnitude in the target's own unit (0 = satisfied)."""
        v = np.asarray(values, dtype=float)
        return np.maximum(0.0, v - self.limit) if self.operator == "<=" else np.maximum(0.0, self.limit - v)

    def describe(self) -> str:
        return f"{self.target} {self.operator} {self.limit:g}"

    def to_dict(self) -> dict:
        return asdict(self)


def bounds_array(bounds: dict[str, tuple[float, float]], features: list[str]) -> np.ndarray:
    missing = [f for f in features if f not in bounds]
    if missing:
        raise ValidationError(f"Missing bounds for: {missing}")
    return np.array([[float(bounds[f][0]), float(bounds[f][1])] for f in features])


def check_bounds(bounds: dict, train_ranges: dict | None = None, allow_extrapolation: bool = False) -> list[str]:
    """Raise ``ValidationError`` for invalid bounds, return warnings otherwise."""
    result = validate_bounds(bounds, train_ranges, allow_extrapolation)
    result.raise_if_invalid()
    return result.warnings


def within_bounds(x, bnds: np.ndarray, tol: float = 1e-9) -> bool:
    x = np.asarray(x, dtype=float)
    return bool(np.all(x >= bnds[:, 0] - tol) and np.all(x <= bnds[:, 1] + tol))


def enforce_bounds(x, bnds: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), bnds[:, 0], bnds[:, 1])
