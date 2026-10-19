"""Headless UI tests of the three dashboard sections with Streamlit's AppTest.

Each test drives the real app (``app.py``) against a temporary database, switching
sections through the top navigation and clicking the real buttons.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(scope="module")
def app_env(tmp_path_factory):
    base = tmp_path_factory.mktemp("ui")
    old = {k: os.environ.get(k) for k in ("METAOPT_DB_PATH", "METAOPT_MODELS_DIR", "METAOPT_REPORTS_DIR",
                                          "METAOPT_UPLOAD_DIR")}
    os.environ.update(METAOPT_DB_PATH=str(base / "ui.db"), METAOPT_MODELS_DIR=str(base / "models"),
                      METAOPT_REPORTS_DIR=str(base / "reports"), METAOPT_UPLOAD_DIR=str(base / "uploads"))
    # config reads the environment at import time; the AppTest script imports it fresh in this process,
    # so reload it to pick up the temporary paths.
    import importlib

    import config
    importlib.reload(config)
    yield base
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    importlib.reload(config)


def _ok(at: AppTest) -> AppTest:
    assert not at.exception, [e.value for e in at.exception]
    return at


def _section(name: str) -> AppTest:
    """Open one section the way app.py does (same theme, same navigation registry)."""
    script = (f"import sys; sys.path.insert(0, {str(Path(APP).parent)!r})\n"
              "import streamlit as st\n"
              "from ui import dataset_model, predict_optimize, results_history\n"
              "from ui.theme import inject_styles\n"
              "inject_styles()\n"
              "fn = {'dataset': dataset_model.render, 'predict': predict_optimize.render,\n"
              "      'results': results_history.render}[%r]\n"
              "st.navigation([st.Page(fn, url_path=%r, default=True)], position='hidden').run()\n" % (name, name))
    return AppTest.from_string(script, default_timeout=600)


def _button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def test_three_sections_and_full_workflow(app_env):
    at = _ok(AppTest.from_file(APP, default_timeout=600).run())
    assert at.title[0].value == "Dataset & Model"
    assert any("No models trained yet" in m.value for m in at.markdown)       # polished empty state

    # Train a quick model set through the real form
    models = next(m for m in at.multiselect if m.label == "Models")
    models.set_value(["dummy", "random_forest", "extra_trees"])
    _ok(_button(at, "Train models").click().run())
    assert any("Model comparison" in m.value for m in at.markdown)
    assert len(at.dataframe) >= 2

    # Predict & Optimize
    at = _ok(_section("predict").run())
    assert at.title[0].value == "Predict & Optimize"
    _ok(_button(at, "Predict performance").click().run())
    labels = [m.label for m in at.metric]
    assert "Predicted S11" in labels and "Predicted Gain" in labels
    _ok(_button(at, "Save prediction to history").click().run())
    assert any("Saved as prediction" in t.value for t in at.toast)
    _ok(_button(at, "Run optimization").click().run())
    assert any(m.label == "Predicted S11" and m.delta for m in at.metric)      # candidate compared with baseline

    # Invalid input is rejected inline and the action is disabled
    at.number_input[0].set_value(-5.0)
    _ok(at.run())
    assert _button(at, "Predict performance").disabled
    assert any("sanity range" in e.value for e in at.error)

    # Results & History
    at = _ok(_section("results").run())
    assert at.title[0].value == "Results & History"
    assert [t.label for t in at.tabs] == ["Model performance", "Explainability", "Optimization history", "Data & export"]
    assert len(at.get("download_button")) >= 4                                 # CSV exports
    _ok(_button(at, "Prepare ZIP of all CSVs").click().run())
    assert any(b.label == "Download ZIP" for b in at.get("download_button"))
