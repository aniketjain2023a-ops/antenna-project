"""End-to-end test of the real website (React frontend + FastAPI backend) in a browser.

Requires both servers to be running and Playwright with Chromium:
    uvicorn backend.main:app --port 8000          (terminal 1)
    cd frontend && npm run dev                     (terminal 2)
    METAOPT_E2E_URL=http://localhost:5173 python -m pytest tests/e2e -v

Skipped automatically when METAOPT_E2E_URL is not set. It uses the backend's real database, so the
METU-SIMPL pipeline (scripts/prepare_public_datasets.py) must have been run.
"""

from __future__ import annotations

import os
import re

import pytest

URL = os.environ.get("METAOPT_E2E_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set METAOPT_E2E_URL to run the browser end-to-end test")


@pytest.fixture(scope="module")
def page():
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        exe = os.environ.get("METAOPT_CHROMIUM")  # optional explicit browser path
        browser = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
        pg = browser.new_page(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
        errors: list[str] = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.errors = errors
        yield pg
        browser.close()


def _expect(locator, timeout=60_000):
    locator.first.wait_for(state="visible", timeout=timeout)
    return locator.first


def test_home_lists_datasets_honestly(page):
    page.goto(URL + "/")
    _expect(page.get_by_role("heading", name="Antenna design, predicted and optimized."))
    _expect(page.get_by_role("button", name=re.compile("METU-SIMPL")))
    _expect(page.get_by_role("button", name=re.compile("Synthetic demo")))
    _expect(page.get_by_text("Not available"))                       # Zenodo: pipeline incomplete
    nav = page.get_by_role("navigation", name="Main").get_by_role("link")
    assert [n.inner_text() for n in nav.all()][-3:] == ["Home", "Design & Optimize", "Results"]


def test_train_models_from_the_website(page):
    page.goto(URL + "/")
    page.get_by_role("button", name=re.compile("Synthetic demo")).click()
    for name in ("Gradient Boosting",):                              # keep the quick test quick
        page.get_by_role("checkbox", name=name).click()
    page.get_by_role("button", name="Train models").click()
    _expect(page.get_by_text(re.compile(r"Trained \d+ models in")), timeout=120_000)
    _expect(page.get_by_text("Latest training run"))


def test_live_prediction_validation_and_save(page):
    page.goto(URL + "/design")
    page.get_by_role("radio", name="METU-SIMPL").click()
    _expect(page.get_by_text("Performance prediction"))
    _expect(page.get_by_text("Within training range"))
    assert re.search(r"-?\d+\.\d+ dB", page.get_by_text("S11 at 28 GHz", exact=True).first.locator("xpath=..").inner_text())
    length = page.get_by_label("Patch length (mm)")
    length.fill("1.0")                                                # below the training range
    _expect(page.get_by_text("Extrapolation · unreliable"))
    length.fill("abc")                                                # not a number
    _expect(page.get_by_text("Enter a number."))
    _expect(page.get_by_text("Fix the inputs"))
    page.get_by_role("button", name="Median").click()
    _expect(page.get_by_text("Within training range"))
    page.get_by_role("button", name="Save prediction").click()
    _expect(page.get_by_text(re.compile(r"Saved as prediction #\d+")))


def test_optimization_on_both_datasets(page):
    page.goto(URL + "/design")
    page.get_by_role("radio", name="METU-SIMPL").click()
    page.get_by_role("button", name="Run optimization").click()
    _expect(page.get_by_text("ML-selected candidate"), timeout=120_000)
    _expect(page.get_by_text("Nearest simulated design"))
    _expect(page.get_by_text("Not physically validated."))
    page.get_by_role("radio", name="Synthetic demo").click()
    _expect(page.get_by_label("Length L (mm)"))                       # schema switches with the dataset
    page.get_by_role("button", name="Run optimization").click()
    _expect(page.get_by_text("Synthetic oracle check"), timeout=120_000)


def test_results_and_csv_download(page, tmp_path):
    page.goto(URL + "/results")
    _expect(page.get_by_text("Test R² by dataset and target"))
    _expect(page.get_by_text("Optimization history"))
    _expect(page.get_by_text(re.compile(r"Actual vs predicted ·")))
    with page.expect_download() as dl:
        page.get_by_role("link", name="Model experiments").click()
    path = tmp_path / "experiments.csv"
    dl.value.save_as(path)
    header = path.read_text().splitlines()[0]
    assert header.startswith("id,dataset_id") and "r2" in header
    assert not page.errors, page.errors
