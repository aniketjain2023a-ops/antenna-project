"""REST API tests (FastAPI TestClient) against a temporary database with the synthetic dataset."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from backend import service  # noqa: E402
from backend.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # conftest points METAOPT_DB_PATH etc. at a temporary directory before anything is imported
    for fn in (service.repo, service._frame, service._load, service.diagnostics):
        fn.cache_clear()
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def synthetic(client):
    ds = next(d for d in client.get("/api/datasets").json()["datasets"] if d["key"] == "synthetic")
    r = client.post(f"/api/datasets/{ds['id']}/train", json={"models": ["dummy", "random_forest"], "cv_folds": 3})
    assert r.status_code == 200, r.text
    return client.get(f"/api/datasets/{ds['id']}").json()


def test_health_and_dataset_listing(client):
    assert client.get("/api/health").json()["status"] == "ok"
    body = client.get("/api/datasets").json()
    keys = {d["key"] for d in body["datasets"]}
    assert "synthetic" in keys
    syn = next(d for d in body["datasets"] if d["key"] == "synthetic")
    assert syn["type"] == "SYNTHETIC" and syn["sample_count"] == 500
    assert [f["name"] for f in syn["features"]] == ["L_mm", "G_mm"]
    assert {t["name"] for t in syn["targets"]} == {"S11_dB", "Gain_dBi"}
    assert set(body["default_models"]) == {"dummy", "random_forest", "gradient_boosting"}
    assert client.get("/api/datasets/9999").status_code == 404


def test_training_selects_models_by_cv(synthetic, client):
    assert synthetic["ready"]
    active = synthetic["active_models"]
    assert set(active) == {"S11_dB", "Gain_dBi"}
    assert all(m["model"] == "random_forest" for m in active.values())          # never the dummy
    exps = client.get(f"/api/datasets/{synthetic['id']}/experiments").json()["experiments"]
    assert len(exps) == 4 and all(e["r2"] is not None for e in exps)
    bad = client.post(f"/api/datasets/{synthetic['id']}/train", json={"models": ["svm"]})
    assert bad.status_code == 422


def test_prediction_valid_extrapolated_and_invalid(synthetic, client):
    url = f"/api/datasets/{synthetic['id']}/predict"
    ok = client.post(url, json={"params": {"L_mm": 10.0, "G_mm": 1.0}}).json()
    assert {p["target"] for p in ok["predictions"]} == {"S11_dB", "Gain_dBi"}
    assert not ok["extrapolation"] and ok["saved_id"] is None
    assert all(p["uncertainty"] is not None for p in ok["predictions"])          # random forest spread
    ext = client.post(url, json={"params": {"L_mm": 14.0, "G_mm": 1.0}}).json()
    assert ext["extrapolation"] and ext["warnings"]
    bad = client.post(url, json={"params": {"L_mm": -1.0, "G_mm": 1.0}})
    assert bad.status_code == 422 and "sanity range" in bad.json()["detail"]
    saved = client.post(url, json={"params": {"L_mm": 10.0, "G_mm": 1.0}, "save": True}).json()
    assert isinstance(saved["saved_id"], int)


def test_optimization_respects_bounds_and_constraint(synthetic, client):
    f = {x["name"]: x for x in synthetic["features"]}
    body = {
        "objective": "Gain_dBi",
        "constraint": {"target": "S11_dB", "operator": "<=", "limit": -10},
        "bounds": {"L_mm": [8.5, 12.0], "G_mm": [0.4, 1.8]},
        "baseline": {"L_mm": f["L_mm"]["median"], "G_mm": f["G_mm"]["median"]},
        "budget": 400,
        "seed": 1,
    }
    r = client.post(f"/api/datasets/{synthetic['id']}/optimize", json=body)
    assert r.status_code == 200, r.text
    res = r.json()
    cand = res["candidate"]
    assert 8.5 <= cand["parameters"]["L_mm"] <= 12.0 and 0.4 <= cand["parameters"]["G_mm"] <= 1.8
    assert cand["feasible"] and cand["predictions"]["S11_dB"] <= -10
    assert res["synthetic_oracle"] and res["history"]
    # holding a parameter fixed: only L varies, G stays at the baseline
    held = client.post(f"/api/datasets/{synthetic['id']}/optimize",
                       json={**body, "bounds": {"L_mm": [8.5, 12.0]}}).json()
    assert held["free_parameters"] == ["L_mm"] and held["candidate"]["parameters"]["G_mm"] == body["baseline"]["G_mm"]
    # bounds outside the training range are rejected, not extrapolated
    out = client.post(f"/api/datasets/{synthetic['id']}/optimize", json={**body, "bounds": {"L_mm": [5.0, 20.0]}})
    assert out.status_code == 422
    assert client.post(f"/api/datasets/{synthetic['id']}/optimize", json={**body, "objective": "L_mm"}).status_code == 422


def test_results_diagnostics_and_csv(synthetic, client):
    s = client.get("/api/results/summary").json()
    assert any(m["dataset_label"] == "Synthetic demo" for m in s["metrics"])
    assert s["optimization_runs"] and s["optimization_runs"][0]["candidate_value"] is not None
    exp_id = next(m["id"] for m in s["metrics"] if m["model"] == "random_forest")
    d = client.get(f"/api/experiments/{exp_id}/diagnostics").json()
    assert len(d["points"]) == 100 and d["importance"]["permutation"] and d["importance"]["impurity"]
    csv = client.get("/api/export/optimizations.csv")
    assert csv.status_code == 200 and "attachment" in csv.headers["content-disposition"]
    assert csv.text.splitlines()[0].startswith("id,")
    assert client.get("/api/export/secrets.csv").status_code == 404
