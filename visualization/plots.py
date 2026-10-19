"""Interactive Plotly figures for the dashboard.

Design rules: every axis carries a unit, categorical colours come from a fixed
validated order (never cycled), sequential maps use a single hue, the
correlation map uses a two-hue diverging scale with a neutral midpoint, and no
chart uses two y-axes. Functions return ``None`` instead of an empty figure
when the required data is missing, so the UI can show an explanation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import label_of

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MUTED = "#8a8985"
SEQUENTIAL = "Blues"
SEQUENTIAL_UNC = "Oranges"
DIVERGING = "RdBu"


def _layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(title=dict(text=title, x=0.0, xanchor="left"), height=height,
                      margin=dict(l=60, r=20, t=60, b=50), legend=dict(orientation="h", y=-0.2))
    return fig


def distribution_grid(df: pd.DataFrame, columns: list[str]) -> go.Figure | None:
    cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()]
    if not cols:
        return None
    n = len(cols)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[label_of(c) for c in cols])
    for i, c in enumerate(cols):
        fig.add_trace(go.Histogram(x=df[c].dropna(), nbinsx=30, marker_color=SERIES[0],
                                   marker_line=dict(width=1, color="white"), name=c, showlegend=False),
                      row=i // ncols + 1, col=i % ncols + 1)
        fig.update_xaxes(title_text=label_of(c), row=i // ncols + 1, col=i % ncols + 1)
        fig.update_yaxes(title_text="count", row=i // ncols + 1, col=i % ncols + 1)
    return _layout(fig, "Distribution of each variable", height=300 * nrows)


def correlation_heatmap(df: pd.DataFrame, columns: list[str]) -> go.Figure | None:
    cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(cols) < 2:
        return None
    corr = df[cols].corr(method="pearson")
    fig = go.Figure(go.Heatmap(z=corr.values, x=cols, y=cols, zmin=-1, zmax=1, colorscale=DIVERGING,
                               text=np.round(corr.values, 2), texttemplate="%{text}",
                               colorbar=dict(title="Pearson r")))
    return _layout(fig, "Pearson correlation matrix (linear association only)", height=380)


def scatter_design_space(df: pd.DataFrame, fx: str, fy: str, color: str | None) -> go.Figure | None:
    if fx not in df or fy not in df:
        return None
    marker = dict(size=8, line=dict(width=1, color="white"))
    if color and color in df:
        marker.update(color=df[color], colorscale=SEQUENTIAL, colorbar=dict(title=label_of(color)), showscale=True)
    else:
        marker.update(color=SERIES[0])
    fig = go.Figure(go.Scatter(x=df[fx], y=df[fy], mode="markers", marker=marker,
                               hovertemplate=f"{fx}=%{{x:.3f}}<br>{fy}=%{{y:.3f}}<extra></extra>"))
    fig.update_xaxes(title=label_of(fx))
    fig.update_yaxes(title=label_of(fy))
    return _layout(fig, f"Sampled design space coloured by {color or 'sample'}")


def actual_vs_predicted(y_true, y_pred, target: str, title_suffix: str = "", y_std=None) -> go.Figure:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    err = dict(type="data", array=1.96 * np.asarray(y_std), color="rgba(42,120,214,0.35)", thickness=1) \
        if y_std is not None else None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", name="test samples", error_y=err,
                             marker=dict(color=SERIES[0], size=8, line=dict(width=1, color="white")),
                             hovertemplate="actual=%{x:.3f}<br>predicted=%{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="ideal y = x",
                             line=dict(color=MUTED, width=2, dash="dash")))
    fig.update_xaxes(title=f"Actual {label_of(target)}")
    fig.update_yaxes(title=f"Predicted {label_of(target)}")
    note = " (error bars: +/-1.96 sigma, approximate)" if y_std is not None else ""
    return _layout(fig, f"Actual vs predicted {target} - held-out test set{title_suffix}{note}")


def residual_plot(y_true, y_pred, target: str, title_suffix: str = "") -> go.Figure:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    resid = y_true - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_pred, y=resid, mode="markers", name="residual",
                             marker=dict(color=SERIES[1], size=8, line=dict(width=1, color="white")),
                             hovertemplate="predicted=%{x:.3f}<br>residual=%{y:.3f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=MUTED, width=2, dash="dash"))
    unit = label_of(target).split("[")[-1].rstrip("]") if "[" in label_of(target) else ""
    fig.update_xaxes(title=f"Predicted {label_of(target)}")
    fig.update_yaxes(title=f"Residual (actual - predicted) [{unit}]" if unit else "Residual")
    return _layout(fig, f"Residuals for {target}{title_suffix}")


def metric_comparison(table: pd.DataFrame, metric: str = "test_R2") -> go.Figure | None:
    if table.empty or metric not in table:
        return None
    t = table.dropna(subset=[metric]).copy()
    if t.empty:
        return None
    t["label"] = t["model"] + " (" + t["variant"] + ")"
    targets = list(dict.fromkeys(t["target"]))
    fig = go.Figure()
    for i, target in enumerate(targets[:8]):
        sub = t[t["target"] == target]
        fig.add_trace(go.Bar(x=sub["label"], y=sub[metric], name=target, marker_color=SERIES[i],
                             hovertemplate="%{x}<br>" + metric + "=%{y:.3f}<extra>" + target + "</extra>"))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
    fig.update_yaxes(title=metric.replace("_", " ") + (" [target unit]" if "R2" not in metric else " [-]"))
    fig.update_xaxes(title="model (variant)")
    return _layout(fig, f"Model comparison: {metric.replace('_', ' ')} per target")


def importance_bar(df: pd.DataFrame | None, value_col: str, title: str, err_col: str | None = None) -> go.Figure | None:
    if df is None or df.empty:
        return None
    d = df.sort_values(value_col)
    fig = go.Figure(go.Bar(
        x=d[value_col], y=d["feature"], orientation="h", marker_color=SERIES[0],
        error_x=dict(type="data", array=d[err_col], color=MUTED) if err_col else None,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.update_xaxes(title=value_col.replace("_", " "))
    fig.update_yaxes(title="input parameter")
    return _layout(fig, title, height=max(260, 80 * len(d) + 140))


def sensitivity_lines(frames: dict[str, pd.DataFrame], feature: str, target: str, base_value: float | None = None) -> go.Figure | None:
    if not frames:
        return None
    fig = go.Figure()
    for i, (name, df) in enumerate(frames.items()):
        color = SERIES[i % len(SERIES)]
        if df["std"].notna().any():
            fig.add_trace(go.Scatter(x=np.concatenate([df[feature], df[feature][::-1]]),
                                     y=np.concatenate([df["prediction"] + df["std"], (df["prediction"] - df["std"])[::-1]]),
                                     fill="toself", fillcolor="rgba(42,120,214,0.15)" if i == 0 else "rgba(235,104,52,0.15)",
                                     line=dict(width=0), name=f"{name} +/-1 sigma (approx.)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=df[feature], y=df["prediction"], mode="lines", name=name,
                                 line=dict(color=color, width=2)))
    if base_value is not None:
        fig.add_vline(x=base_value, line=dict(color=MUTED, dash="dot"), annotation_text="base point")
    fig.update_xaxes(title=label_of(feature))
    fig.update_yaxes(title=f"Predicted {label_of(target)}")
    return _layout(fig, f"One-at-a-time sensitivity of {target} to {feature}")


def partial_dependence_line(df: pd.DataFrame, feature: str, target: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=df[feature], y=df["partial_dependence"], mode="lines",
                               line=dict(color=SERIES[0], width=2), name="partial dependence"))
    fig.update_xaxes(title=label_of(feature))
    fig.update_yaxes(title=f"Average predicted {label_of(target)}")
    return _layout(fig, f"Partial dependence of {target} on {feature}")


def design_space_map(grid: dict, values: np.ndarray, title: str, colorbar: str, scale: str = SEQUENTIAL,
                     points: pd.DataFrame | None = None, markers: pd.DataFrame | None = None,
                     reverse: bool = False) -> go.Figure:
    fig = go.Figure(go.Heatmap(x=grid["x"], y=grid["y"], z=values, colorscale=scale, reversescale=reverse,
                               colorbar=dict(title=colorbar),
                               hovertemplate=f"{grid['fx']}=%{{x:.3f}}<br>{grid['fy']}=%{{y:.3f}}<br>value=%{{z:.3f}}<extra></extra>"))
    if points is not None and len(points):
        fig.add_trace(go.Scatter(x=points[grid["fx"]], y=points[grid["fy"]], mode="markers", name="training samples",
                                 marker=dict(size=5, color="rgba(255,255,255,0.8)", line=dict(width=1, color="#333"))))
    if markers is not None and len(markers):
        labels = markers["label"] if "label" in markers else None
        # Text labels only when points are well separated; otherwise rely on hover to avoid overlaps.
        spread = np.ptp(markers[grid["fx"]]) / max(grid["x"][-1] - grid["x"][0], 1e-12) if len(markers) > 1 else 1
        fig.add_trace(go.Scatter(x=markers[grid["fx"]], y=markers[grid["fy"]],
                                 mode="markers+text" if labels is not None and spread > 0.15 else "markers",
                                 text=labels, textposition="top center", name="highlighted",
                                 hovertemplate="%{text}<br>%{x:.3f}, %{y:.3f}<extra></extra>",
                                 marker=dict(size=13, symbol="x", color=SERIES[1], line=dict(width=2, color="white"))))
    fig.update_xaxes(title=label_of(grid["fx"]))
    fig.update_yaxes(title=label_of(grid["fy"]))
    return _layout(fig, title, height=460)


def baseline_vs_candidates(rows: pd.DataFrame, target: str) -> go.Figure | None:
    """Bar chart of one target for baseline and each candidate (one chart per target: no dual axes)."""
    if rows.empty or target not in rows:
        return None
    std_col = f"{target}_std"
    err = dict(type="data", array=rows[std_col].fillna(0), color=MUTED) if std_col in rows and rows[std_col].notna().any() else None
    colors = [MUTED if s == "baseline" else SERIES[0] for s in rows["source"]]
    fig = go.Figure(go.Bar(x=rows["source"], y=rows[target], marker_color=colors, error_y=err,
                           hovertemplate="%{x}<br>%{y:.3f}<extra></extra>"))
    fig.update_yaxes(title=f"Predicted {label_of(target)}")
    fig.update_xaxes(title="design")
    return _layout(fig, f"Baseline vs ML-selected candidates: predicted {target}", height=360)


def tradeoff_scatter(samples: pd.DataFrame, x_target: str, y_target: str,
                     candidates: pd.DataFrame | None = None, baseline: dict | None = None) -> go.Figure | None:
    if x_target not in samples or y_target not in samples:
        return None
    fig = go.Figure()
    infeas = samples[~samples["feasible"]]
    feas = samples[samples["feasible"]]
    fig.add_trace(go.Scatter(x=infeas[x_target], y=infeas[y_target], mode="markers", name="infeasible samples",
                             marker=dict(size=6, color="rgba(138,137,133,0.35)")))
    fig.add_trace(go.Scatter(x=feas[x_target], y=feas[y_target], mode="markers", name="feasible samples",
                             marker=dict(size=6, color="rgba(42,120,214,0.45)")))
    front = samples[samples["pareto"]].sort_values(x_target)
    if len(front):
        fig.add_trace(go.Scatter(x=front[x_target], y=front[y_target], mode="lines+markers", name="Pareto front (predicted)",
                                 line=dict(color=SERIES[2], width=2), marker=dict(size=8)))
    if candidates is not None and len(candidates):
        fig.add_trace(go.Scatter(x=candidates[x_target], y=candidates[y_target], mode="markers",
                                 text=candidates["source"], name="optimizer candidates",
                                 hovertemplate="%{text}<br>S11/x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
                                 marker=dict(size=14, symbol="star", color=SERIES[1], line=dict(width=1, color="white"))))
    if baseline:
        fig.add_trace(go.Scatter(x=[baseline[x_target]], y=[baseline[y_target]], mode="markers+text", text=["baseline"],
                                 textposition="bottom center", name="baseline",
                                 marker=dict(size=13, symbol="diamond", color="#333")))
    fig.update_xaxes(title=f"Predicted {label_of(x_target)}")
    fig.update_yaxes(title=f"Predicted {label_of(y_target)}")
    return _layout(fig, "Predicted trade-off between targets (surrogate model, not measured)", height=480)


def convergence_plot(histories: dict[str, list[float]]) -> go.Figure | None:
    histories = {k: v for k, v in histories.items() if v}
    if not histories:
        return None
    fig = go.Figure()
    for i, (name, h) in enumerate(histories.items()):
        fig.add_trace(go.Scatter(x=np.arange(1, len(h) + 1), y=h, mode="lines", name=name,
                                 line=dict(color=SERIES[i % len(SERIES)], width=2)))
    finals = [min(h) for h in histories.values()]
    upper = max(np.quantile(h, 0.9) for h in histories.values())
    lo = min(finals)
    pad = 0.05 * max(upper - lo, 1e-6)
    fig.update_yaxes(range=[lo - pad, upper + pad])
    fig.update_xaxes(title="surrogate evaluations")
    fig.update_yaxes(title="best penalised objective so far [-] (lower is better)")
    return _layout(fig, "Optimizer convergence")
