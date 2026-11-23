"""Unit tests for the detection pipeline's logic (no dataset needed)."""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_detection as ed  # noqa: E402
from prepare_data import MEASURED  # noqa: E402


def frame(rows: list[dict]) -> pl.DataFrame:
    """Patient-hour rows with every measured column present (None unless given)."""
    base = {c: None for c in MEASURED} | {f"{c}__obs": False for c in MEASURED}
    return pl.DataFrame([base | r for r in rows], infer_schema_length=None)


def test_clinical_score_counts_criteria_and_treats_missing_as_not_met():
    df = frame([
        {"HR": 120.0, "Temp": 39.0, "Resp": 25.0, "PaCO2": 40.0, "WBC": 15.0, "SBP": 90.0},  # all 6 met
        {"HR": 70.0, "Temp": 37.0, "Resp": 14.0, "PaCO2": 40.0, "WBC": 8.0, "SBP": 120.0},  # none met
        {},  # all missing
    ])
    assert ed.clinical_score(df).tolist() == [6.0, 0.0, 0.0]


def test_guardrails_block_missing_map_or_hr_and_sparse_records():
    full = {c: 1.0 for c in MEASURED} | {f"{c}__obs": True for c in MEASURED}
    no_map = full | {"MAP": None, "MAP__obs": False}
    sparse = {"MAP": 70.0, "HR": 80.0, "MAP__obs": True, "HR__obs": True}  # 2 of 33 present
    stale = {c: 1.0 for c in MEASURED}  # carried forward, nothing measured this hour
    df = frame([full, no_map, sparse, stale])
    assert ed.ethos_allowed(df, "last_known").tolist() == [True, False, False, True]
    assert ed.ethos_allowed(df, "this_hour").tolist() == [True, False, False, False]


def test_alerting_lead_time_uses_onset_six_hours_after_first_label():
    # Septic patient: labels from hour 4 -> onset at hour 10; first alert at hour 3 -> 7 h lead.
    # Clean patient: never above threshold -> specificity 1.
    rows = [{"Patient_ID": 1, "Hour": h, "SepsisLabel": int(h >= 4), "ever_septic": 1} for h in range(8)]
    rows += [{"Patient_ID": 2, "Hour": h, "SepsisLabel": 0, "ever_septic": 0} for h in range(8)]
    df = pl.DataFrame(rows)
    scores = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9] + [0.2] * 8)
    out = ed.alerting(df, scores, threshold=0.5)
    assert out["median_lead_hours_when_early"] == 7
    assert out["patient_sensitivity_before_onset"] == 1.0
    assert out["patient_specificity"] == 1.0


def test_threshold_for_specificity_matches_clean_patient_quantile():
    df = pl.DataFrame({"Patient_ID": [1, 2, 3, 4, 5], "ever_septic": [0, 0, 0, 0, 1]})
    thr = ed.threshold_for_specificity(df, np.array([0.1, 0.2, 0.3, 0.4, 0.9]), 0.75)
    assert thr == pytest.approx(np.quantile([0.1, 0.2, 0.3, 0.4], 0.75))


def test_patient_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(300), 5)
    y = np.repeat(rng.integers(0, 2, 300), 5)
    s = y + rng.normal(0, 1.0, y.size)
    lo, hi = ed.patient_bootstrap(y, s, groups, roc_auc_score, n_boot=100)
    assert lo < roc_auc_score(y, s) < hi
