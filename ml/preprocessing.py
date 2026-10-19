"""Data quality inspection and non-destructive cleaning.

Rules enforced here:
* the source DataFrame is never modified (all functions work on copies);
* missing *target* values are never imputed - rows are excluded per target;
* outliers are only *flagged* (IQR rule), never removed automatically;
* every cleaning action is recorded in a human-readable summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class QualityReport:
    n_rows: int
    n_columns: int
    dtypes: dict[str, str]
    numeric_columns: list[str]
    non_numeric_columns: list[str]
    missing_per_column: dict[str, int]
    duplicate_rows: int
    outliers_per_column: dict[str, int]
    constant_columns: list[str]
    warnings: list[str] = field(default_factory=list)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "dtype": self.dtypes,
                "missing": self.missing_per_column,
                "iqr_outliers": {c: self.outliers_per_column.get(c, 0) for c in self.dtypes},
                "numeric": {c: c in self.numeric_columns for c in self.dtypes},
            }
        )


def detect_column_types(df: pd.DataFrame) -> dict[str, str]:
    """Classify columns as 'numeric', 'numeric-like text', 'categorical/text' or 'empty'."""
    kinds = {}
    for col in df.columns:
        series = df[col]
        if series.dropna().empty:
            kinds[col] = "empty"
        elif pd.api.types.is_numeric_dtype(series):
            kinds[col] = "numeric"
        else:
            coerced = pd.to_numeric(series.dropna(), errors="coerce")
            kinds[col] = "numeric-like text" if coerced.notna().mean() > 0.95 else "categorical/text"
    return kinds


def iqr_outlier_mask(series: pd.Series, k: float = 1.5) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return pd.Series(False, index=series.index)
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


def inspect_quality(df: pd.DataFrame, columns: list[str] | None = None) -> QualityReport:
    cols = list(columns) if columns else list(df.columns)
    sub = df[cols]
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(sub[c])]
    non_numeric = [c for c in cols if c not in numeric]
    missing = {c: int(sub[c].isna().sum()) for c in cols}
    outliers = {c: int(iqr_outlier_mask(sub[c]).sum()) for c in numeric}
    constant = [c for c in numeric if sub[c].nunique(dropna=True) <= 1]
    warnings = []
    if sum(missing.values()):
        warnings.append(f"{sum(missing.values())} missing values found; they will NOT be imputed.")
    dup = int(sub.duplicated().sum())
    if dup:
        warnings.append(f"{dup} fully duplicated rows found.")
    if constant:
        warnings.append(f"Constant columns carry no information: {constant}.")
    if non_numeric:
        warnings.append(f"Non-numeric columns cannot be used as model inputs/targets: {non_numeric}.")
    return QualityReport(
        n_rows=len(sub),
        n_columns=len(cols),
        dtypes={c: str(sub[c].dtype) for c in cols},
        numeric_columns=numeric,
        non_numeric_columns=non_numeric,
        missing_per_column=missing,
        duplicate_rows=dup,
        outliers_per_column=outliers,
        constant_columns=constant,
        warnings=warnings,
    )


@dataclass
class CleaningSummary:
    rows_in: int
    rows_out: int
    actions: list[str]

    def as_text(self) -> str:
        return "\n".join([f"Rows in: {self.rows_in}", f"Rows out: {self.rows_out}", *self.actions])


def clean_dataframe(
    df: pd.DataFrame,
    features: list[str],
    targets: list[str],
    rename: dict[str, str] | None = None,
    drop_missing_features: bool = True,
    drop_rows_without_any_target: bool = True,
    drop_duplicates: bool = True,
    coerce_numeric: bool = True,
) -> tuple[pd.DataFrame, CleaningSummary]:
    """Return a cleaned COPY containing only the selected columns.

    Missing target values are kept as NaN (rows are excluded per-target at
    training time) unless a row has no target at all.
    """
    if not features:
        raise ValueError("Select at least one input feature.")
    overlap = set(features) & set(targets)
    if overlap:
        raise ValueError(f"Columns cannot be both input and target: {sorted(overlap)}")
    missing_cols = [c for c in features + targets if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found: {missing_cols}")

    out = df[features + targets].copy()
    actions: list[str] = [f"Selected {len(features)} input feature(s) and {len(targets)} target(s)."]
    rows_in = len(out)

    if coerce_numeric:
        for col in features + targets:
            if not pd.api.types.is_numeric_dtype(out[col]):
                before = out[col].isna().sum()
                out[col] = pd.to_numeric(out[col], errors="coerce")
                added = int(out[col].isna().sum() - before)
                actions.append(f"Converted '{col}' to numeric ({added} unparsable value(s) became missing).")
    if drop_missing_features:
        mask = out[features].isna().any(axis=1)
        if mask.any():
            out = out.loc[~mask]
            actions.append(f"Dropped {int(mask.sum())} row(s) with missing input features.")
    if targets and drop_rows_without_any_target:
        mask = out[targets].isna().all(axis=1)
        if mask.any():
            out = out.loc[~mask]
            actions.append(f"Dropped {int(mask.sum())} row(s) with no target values.")
    for t in targets:
        n_missing = int(out[t].isna().sum())
        if n_missing:
            actions.append(f"Target '{t}' has {n_missing} missing value(s): kept as missing (no imputation).")
    if drop_duplicates:
        dup = out.duplicated()
        if dup.any():
            out = out.loc[~dup]
            actions.append(f"Removed {int(dup.sum())} exact duplicate row(s).")
    if rename:
        rename = {k: v for k, v in rename.items() if v and k != v}
        if rename:
            out = out.rename(columns=rename)
            actions.append(f"Renamed columns: {rename}.")
    out = out.reset_index(drop=True)
    actions.append("Outliers were flagged for inspection only; none were removed.")
    return out, CleaningSummary(rows_in=rows_in, rows_out=len(out), actions=actions)


def feature_ranges(df: pd.DataFrame, features: list[str]) -> dict[str, tuple[float, float]]:
    return {f: (float(df[f].min()), float(df[f].max())) for f in features}


def xy_for_target(df: pd.DataFrame, features: list[str], target: str) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Return X, y for rows where the target and all features are present."""
    if target not in df.columns:
        raise KeyError(f"Target '{target}' is not present in the dataset.")
    sub = df[features + [target]].apply(pd.to_numeric, errors="coerce").dropna()
    return sub[features].to_numpy(dtype=float), sub[target].to_numpy(dtype=float), sub.index
