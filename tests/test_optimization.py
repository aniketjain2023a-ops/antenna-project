"""Surrogate optimization: objectives, constraints, bounds, algorithms and persistence."""

from __future__ import annotations

import numpy as np
import pytest

from optimization import ALGORITHMS
from optimization.analysis import pareto_mask, rank_candidates, sample_design_space
from optimization.constraints import Constraint, bounds_array, check_bounds, within_bounds
from optimization.objectives import ObjectiveConfig, SurrogateEvaluator, evaluate_objective
from services import workflow
from utils.validation import ValidationError


@pytest.fixture(scope="module")
def setup(trained):
    models = trained["models"]
    df, _, _ = workflow.load_dataset_frame(trained["repo"], trained["dataset"]["id"])
    norm = workflow.target_ranges(df, list(models))
    bounds = workflow.model_training_ranges(models)
    return {**trained, "norm": norm, "bounds": bounds}


def test_constraint_violation_math():
    c = Constraint("S11_dB", "<=", -10.0)
    assert np.allclose(c.violation([-12.0, -10.0, -7.0]), [0.0, 0.0, 3.0])
    g = Constraint("Gain_dBi", ">=", 6.0)
    assert np.allclose(g.violation([7.0, 5.5]), [0.0, 0.5])
    with pytest.raises(ValueError):
        Constraint("S11_dB", "==", 1.0)


def test_invalid_bounds_rejected(setup):
    with pytest.raises(ValidationError):
        check_bounds({"L_mm": (12.0, 9.0), "G_mm": (0.5, 1.5)})
    with pytest.raises(ValidationError):
        check_bounds({"L_mm": (5.0, 15.0), "G_mm": (0.5, 1.5)}, setup["bounds"])   # extrapolation not allowed
    assert check_bounds({"L_mm": (5.0, 15.0), "G_mm": (0.5, 1.5)}, setup["bounds"], allow_extrapolation=True)
    with pytest.raises(ValidationError):
        bounds_array({"L_mm": (1, 2)}, ["L_mm", "G_mm"])


def test_objective_normalization_and_penalty(setup):
    cfg = ObjectiveConfig("tradeoff", [Constraint("S11_dB", "<=", -10.0)], {"S11_dB": 1, "Gain_dBi": 1}, setup["norm"])
    s_lo, s_hi = setup["norm"]["S11_dB"]
    g_lo, g_hi = setup["norm"]["Gain_dBi"]
    preds = {"S11_dB": (np.array([s_lo, s_hi]), None), "Gain_dBi": (np.array([g_hi, g_lo]), None)}
    ev = evaluate_objective(cfg, preds)
    assert ev["raw"][0] == pytest.approx(-1.0) and ev["raw"][1] == pytest.approx(0.0)   # best / worst desirability
    assert ev["violation"][0] == 0 and ev["violation"][1] > 0
    assert ev["penalised"][1] > ev["raw"][1]
    with pytest.raises(ValueError):
        ObjectiveConfig("tradeoff", weights={"Bandwidth_MHz": 1.0}, normalization=setup["norm"]).validate(["S11_dB", "Gain_dBi"])
    with pytest.raises(ValueError):
        ObjectiveConfig("max_gain").validate(["S11_dB"])          # no gain model -> cannot maximise gain


@pytest.mark.parametrize("algo", list(ALGORITHMS))
def test_algorithms_respect_bounds_and_constraints(setup, algo):
    ev = SurrogateEvaluator(setup["models"])
    cfg = ObjectiveConfig("max_gain", [Constraint("S11_dB", "<=", -10.0)], normalization=setup["norm"])
    bounds = {"L_mm": (8.5, 12.0), "G_mm": (0.4, 1.8)}
    out = ALGORITHMS[algo][1](ev, cfg, bounds, seed=1, budget=600)
    x = np.array([out.candidate["parameters"][f] for f in ev.features])
    assert within_bounds(x, bounds_array(bounds, ev.features)) and out.within_bounds and not out.bounds_clipped
    assert out.candidate["feasible"] and out.candidate["predictions"]["S11_dB"] <= -10.0
    assert out.n_evals <= 600 * 1.15
    assert out.candidate["predictions"]["Gain_dBi"] > 7.0       # optimizer finds the high-gain region
    assert out.history == sorted(out.history, reverse=True)     # best-so-far is monotone


def test_optimizer_is_reproducible(setup):
    cfg = ObjectiveConfig("min_s11", [Constraint("Gain_dBi", ">=", 6.0)], normalization=setup["norm"])
    runs = [ALGORITHMS["differential_evolution"][1](SurrogateEvaluator(setup["models"]), cfg, setup["bounds"], seed=5, budget=400)
            for _ in range(2)]
    assert runs[0].candidate["parameters"] == runs[1].candidate["parameters"]


def test_unachievable_constraint_reported_infeasible(setup):
    cfg = ObjectiveConfig("max_gain", [Constraint("S11_dB", "<=", -80.0)], normalization=setup["norm"])
    out = ALGORITHMS["differential_evolution"][1](SurrogateEvaluator(setup["models"]), cfg, setup["bounds"], seed=1, budget=300)
    assert not out.candidate["feasible"] and out.candidate["violations"]["S11_dB <= -80"] > 0


def test_run_optimization_persists_and_compares_baseline(setup):
    repo = setup["repo"]
    cfg = ObjectiveConfig("max_gain", [Constraint("S11_dB", "<=", -10.0)], normalization=setup["norm"])
    before = repo.counts()["optimizations"]
    outs = workflow.run_optimization(repo, setup["models"], cfg, setup["bounds"], {"L_mm": 11.5, "G_mm": 0.5},
                                     ["differential_evolution", "multistart_nelder_mead"], seed=3, budget=400)
    assert repo.counts()["optimizations"] == before + 2
    outcome, record, base = outs[0]
    assert record["feasible"] is True and record["within_bounds"] is True
    assert record["baseline_design"] == {"L_mm": 11.5, "G_mm": 0.5}
    assert record["candidate_predictions"]["Gain_dBi"] > record["baseline_predictions"]["Gain_dBi"]
    assert record["objective_value"] < record["baseline_objective"]
    assert record["data_source_type"] == "SYNTHETIC" and record["dataset_checksum"]
    assert set(record["model_experiment_ids"]) == {"S11_dB", "Gain_dBi"}
    assert record["objective_config"]["algorithm_settings"]
    with pytest.raises(ValueError):
        workflow.run_optimization(repo, setup["models"], cfg, setup["bounds"], {"L_mm": 11.5, "G_mm": 0.5}, ["bogus"])
    with pytest.raises(ValidationError):
        workflow.run_optimization(repo, setup["models"], cfg, {"L_mm": (5, 20), "G_mm": (0.3, 1.0)},
                                  {"L_mm": 11.5, "G_mm": 0.5}, ["differential_evolution"])
    ranked = rank_candidates([{**base, "source": "baseline"}, {**outcome.candidate, "source": "de"}], cfg)
    assert ranked.iloc[0]["source"] == "de" and "feasible" in ranked.iloc[0]["explanation"]
    assert "INFEASIBLE" in ranked.iloc[1]["explanation"]


def test_pareto_and_design_space_sampling(setup):
    vals = np.array([[1.0, 1.0], [2.0, 2.0], [0.5, 0.5], [3.0, 0.1]])
    # minimise column 0, maximise column 1
    assert pareto_mask(vals, [True, False]).tolist() == [True, True, True, False]
    cfg = ObjectiveConfig("tradeoff", [Constraint("S11_dB", "<=", -10.0)], {"S11_dB": 0.5, "Gain_dBi": 0.5}, setup["norm"])
    samples = sample_design_space(SurrogateEvaluator(setup["models"]), cfg, setup["bounds"], n=300)
    assert len(samples) == 300 and samples["pareto"].any() and samples["feasible"].any()
    lo, hi = setup["bounds"]["L_mm"]
    assert samples["L_mm"].between(lo, hi).all()
