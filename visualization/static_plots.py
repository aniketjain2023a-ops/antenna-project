"""Matplotlib (static PNG) figures for the PDF report. Headless 'Agg' backend."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import label_of  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MUTED = "#8a8985"


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def actual_vs_predicted_png(y_true, y_pred, target: str, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_true, y_pred, s=18, color=SERIES[0], edgecolor="white", linewidth=0.5, label="test samples")
    lo, hi = float(min(np.min(y_true), np.min(y_pred))), float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "--", color=MUTED, lw=1.5, label="ideal y = x")
    ax.set_xlabel(f"Actual {label_of(target)}")
    ax.set_ylabel(f"Predicted {label_of(target)}")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, path)


def metric_bars_png(table: pd.DataFrame, metric: str, path: Path) -> Path | None:
    t = table.dropna(subset=[metric])
    if t.empty:
        return None
    targets = list(dict.fromkeys(t["target"]))
    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 3.8), squeeze=False)
    for ax, target in zip(axes[0], targets):
        sub = t[t["target"] == target]
        labels = sub["model"] + "\n(" + sub["variant"] + ")"
        ax.bar(labels, sub[metric], color=SERIES[0], edgecolor="white")
        ax.set_title(f"{metric} - {target}", fontsize=9)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", labelsize=7, rotation=45)
        ax.grid(axis="y", alpha=0.25)
    return _save(fig, path)


def importance_png(df: pd.DataFrame, value_col: str, title: str, path: Path) -> Path | None:
    if df is None or df.empty:
        return None
    d = df.sort_values(value_col)
    fig, ax = plt.subplots(figsize=(5, 2 + 0.4 * len(d)))
    ax.barh(d["feature"], d[value_col], color=SERIES[0])
    ax.set_xlabel(value_col.replace("_", " "))
    ax.set_title(title, fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, path)


def heatmap_png(grid: dict, values, title: str, cbar: str, path: Path, cmap: str = "Blues",
                points: pd.DataFrame | None = None, markers: pd.DataFrame | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    im = ax.pcolormesh(grid["x"], grid["y"], values, cmap=cmap, shading="auto")
    fig.colorbar(im, ax=ax, label=cbar)
    if points is not None and len(points):
        ax.scatter(points[grid["fx"]], points[grid["fy"]], s=5, c="white", edgecolor="#333", linewidth=0.4,
                   label="training samples")
    if markers is not None and len(markers):
        ax.scatter(markers[grid["fx"]], markers[grid["fy"]], s=80, marker="x", c=SERIES[1], linewidth=2,
                   label="candidates")
    ax.set_xlabel(label_of(grid["fx"]))
    ax.set_ylabel(label_of(grid["fy"]))
    ax.set_title(title, fontsize=9)
    if (points is not None and len(points)) or (markers is not None and len(markers)):
        ax.legend(fontsize=7, loc="upper right")
    return _save(fig, path)


def baseline_candidate_png(rows: pd.DataFrame, targets: list[str], path: Path) -> Path | None:
    targets = [t for t in targets if t in rows]
    if rows.empty or not targets:
        return None
    fig, axes = plt.subplots(1, len(targets), figsize=(4.5 * len(targets), 3.6), squeeze=False)
    for ax, t in zip(axes[0], targets):
        colors = [MUTED if s == "baseline" else SERIES[0] for s in rows["source"]]
        ax.bar(rows["source"], rows[t], color=colors)
        ax.set_ylabel(f"Predicted {label_of(t)}")
        ax.set_title(f"Baseline vs candidates: {t}", fontsize=9)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        ax.grid(axis="y", alpha=0.25)
    return _save(fig, path)
