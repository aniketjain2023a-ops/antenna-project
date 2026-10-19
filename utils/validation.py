"""Input validation helpers shared by the ML, optimization and UI layers.

Validation never mutates its inputs; it returns errors/warnings so callers can
decide how to react (reject in the UI, raise in library code).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

import config


class ValidationError(ValueError):
    """Raised when user-supplied values are invalid."""


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_invalid(self) -> None:
        if self.errors:
            raise ValidationError("; ".join(self.errors))


def is_finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_geometry(
    params: Mapping[str, float],
    required: list[str],
    train_ranges: Mapping[str, tuple[float, float]] | None = None,
) -> ValidationResult:
    """Validate geometry inputs for prediction.

    * every required feature must be present and finite;
    * millimetre quantities must lie inside the physical sanity limits;
    * values outside the training range produce an *extrapolation warning*.
    """
    result = ValidationResult()
    lo_sane, hi_sane = config.GEOMETRY_SANITY_LIMITS
    for name in required:
        if name not in params:
            result.errors.append(f"Missing parameter '{name}'.")
            continue
        value = params[name]
        if not is_finite_number(value):
            result.errors.append(f"Parameter '{name}' must be a finite number (got {value!r}).")
            continue
        value = float(value)
        lo_name = config.KNOWN_COLUMNS.get(name, {}).get("min_valid", lo_sane)
        if name.endswith("_mm") and not (lo_name <= value <= hi_sane):
            result.errors.append(
                f"Parameter '{name}'={value} mm is outside the physical sanity range "
                f"[{lo_name}, {hi_sane}] mm."
            )
            continue
        if not name.endswith("_mm") and value < 0 and name in config.KNOWN_COLUMNS:
            result.errors.append(f"Parameter '{name}'={value} must not be negative.")
            continue
        if train_ranges and name in train_ranges:
            t_lo, t_hi = train_ranges[name]
            if value < t_lo or value > t_hi:
                result.warnings.append(
                    f"'{name}'={value:g} lies outside the training range "
                    f"[{t_lo:g}, {t_hi:g}]: this is EXTRAPOLATION and the prediction is unreliable."
                )
    return result


def validate_bounds(
    bounds: Mapping[str, tuple[float, float]],
    train_ranges: Mapping[str, tuple[float, float]] | None = None,
    allow_extrapolation: bool = False,
) -> ValidationResult:
    """Validate optimization parameter bounds."""
    result = ValidationResult()
    if not bounds:
        result.errors.append("At least one parameter bound is required.")
    lo_sane, hi_sane = config.GEOMETRY_SANITY_LIMITS
    for name, pair in bounds.items():
        try:
            lo, hi = pair
        except (TypeError, ValueError):
            result.errors.append(f"Bounds for '{name}' must be a (lower, upper) pair.")
            continue
        if not (is_finite_number(lo) and is_finite_number(hi)):
            result.errors.append(f"Bounds for '{name}' must be finite numbers.")
            continue
        lo, hi = float(lo), float(hi)
        if lo >= hi:
            result.errors.append(f"Lower bound of '{name}' ({lo}) must be < upper bound ({hi}).")
            continue
        if name.endswith("_mm") and (lo < config.KNOWN_COLUMNS.get(name, {}).get("min_valid", lo_sane) or hi > hi_sane):
            result.errors.append(f"Bounds of '{name}' violate physical sanity limits [{lo_sane}, {hi_sane}] mm.")
        if train_ranges and name in train_ranges:
            t_lo, t_hi = train_ranges[name]
            if lo < t_lo - 1e-9 or hi > t_hi + 1e-9:
                msg = (f"Bounds of '{name}' [{lo:g}, {hi:g}] extend beyond the training range "
                       f"[{t_lo:g}, {t_hi:g}] (extrapolation).")
                (result.warnings if allow_extrapolation else result.errors).append(msg)
    return result


def dataframe_checksum(df: pd.DataFrame) -> str:
    """Deterministic SHA-256 of a DataFrame's CSV serialisation."""
    payload = df.to_csv(index=False, float_format="%.10g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_checksum(path: str | Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def to_jsonable(value):
    """Convert numpy / pandas scalars and containers into JSON-serialisable values."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)
