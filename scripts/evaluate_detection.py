"""Sepsis early-warning pipeline and evaluation (hospital A -> hospital B).

Stage 3 of the project (after scripts/prepare_data.py and the Rust causal discovery):

1. Detector: LightGBM on the causal-driver features chosen by mRMR on hospital A
   (results/causal_discovery.json, physiological run), scoring every patient-hour.
2. Guardrails: the Effect Ethos rules (ETHOS-001 MAP+HR present, ETHOS-002 <= 50% missing),
   mirrored from backend/src/ethos/mod.rs, block predictions on records too sparse to trust.
3. Explanations: per-prediction feature contributions (LightGBM SHAP values) plus the guardrail
   verdict, written as example diagnostic reports.
4. Evaluation on hospital B against baselines: ICU length of stay alone, a SIRS/qSOFA clinical
   score, mRMR features including time/admin fields, and all features with and without them.

Labels are the challenge's hourly ``SepsisLabel`` (positive from 6 h before clinical onset).

    /usr/bin/python3 scripts/evaluate_detection.py --split data/split --discovery results/causal_discovery.json --out results

LightGBM runs on CPU by default: on the Intel Arc iGPU its OpenCL learner measured about 2x slower
(300 rounds on 670k x 40: 23.0 s GPU vs 11.8 s CPU), so ``--device gpu`` is opt-in.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import LABS, MEASURED, VITALS  # noqa: E402

STATIC = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]
TIME_ADMIN = {"ICULOS", "HospAdmTime", "Unit1", "Unit2"}
ALL_FEATURES = [*VITALS, *LABS, *STATIC, "ICULOS"]
TARGET_PATIENT_SPECIFICITY = (
    0.80  # alert threshold chosen on hospital A validation patients
)

LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "metric": "auc",
    "verbose": -1,
    "seed": 0,
}


@dataclass(frozen=True)
class Result:
    model: str
    features: list[str]
    hourly_auroc: float
    hourly_auroc_ci: tuple[float, float]
    hourly_auprc: float
    hourly_auprc_ci: tuple[float, float]
    patient_auroc: float
    patient_auroc_ci: tuple[float, float]
    train_seconds: float
    rounds: int


# --- detectors --------------------------------------------------------------------------------


def clinical_score(df: pl.DataFrame) -> np.ndarray:
    """SIRS count (0-4) + partial qSOFA (0-2, no mentation), on last-known values; missing = not met."""
    crit = [
        pl.col("HR") > 90,
        (pl.col("Temp") > 38) | (pl.col("Temp") < 36),
        (pl.col("Resp") > 20) | (pl.col("PaCO2") < 32),
        (pl.col("WBC") > 12) | (pl.col("WBC") < 4),
        pl.col("Resp") >= 22,
        pl.col("SBP") <= 100,
    ]
    return (
        df.select(pl.sum_horizontal(c.fill_null(False).cast(pl.Int8) for c in crit))
        .to_series()
        .to_numpy()
        .astype(float)
    )


def split_validation(
    train: pl.DataFrame, seed: int = 0
) -> tuple[pl.DataFrame, pl.DataFrame]:
    val_ids = (
        train.get_column("Patient_ID").unique().sort().sample(fraction=0.15, seed=seed)
    )
    return train.filter(~pl.col("Patient_ID").is_in(val_ids)), train.filter(
        pl.col("Patient_ID").is_in(val_ids)
    )


def fit_lgb(
    fit: pl.DataFrame, val: pl.DataFrame, features: Sequence[str], device: str
) -> tuple[lgb.Booster, float]:
    ds = lambda d: lgb.Dataset(
        d.select(features).to_numpy(),
        d.get_column("SepsisLabel").to_numpy(),
        feature_name=list(features),
    )
    start = time.perf_counter()
    booster = lgb.train(
        {**LGB_PARAMS, "device_type": device},
        ds(fit),
        num_boost_round=2000,
        valid_sets=[ds(val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return booster, time.perf_counter() - start


# --- metrics ----------------------------------------------------------------------------------


def patient_bootstrap(
    y: np.ndarray,
    s: np.ndarray,
    groups: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int,
    seed: int = 0,
) -> tuple[float, float]:
    """95% CI resampling whole patients, so correlated hours within a stay are not treated as independent."""
    rng = np.random.default_rng(seed)
    uniq, inverse = np.unique(groups, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[order], np.arange(len(uniq) + 1))
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate(
            [
                order[bounds[p] : bounds[p + 1]]
                for p in rng.integers(0, len(uniq), len(uniq))
            ]
        )
        if y[idx].min() != y[idx].max():
            stats.append(metric(y[idx], s[idx]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def per_patient_max(df: pl.DataFrame, scores: np.ndarray) -> pl.DataFrame:
    return (
        df.select("Patient_ID", "ever_septic")
        .with_columns(pl.Series("s", scores))
        .group_by("Patient_ID")
        .agg(pl.col("ever_septic").first(), pl.col("s").max())
    )


def evaluate(
    name: str,
    feats: list[str],
    test: pl.DataFrame,
    scores: np.ndarray,
    secs: float,
    rounds: int,
    n_boot: int,
) -> Result:
    y, g = (
        test.get_column("SepsisLabel").to_numpy(),
        test.get_column("Patient_ID").to_numpy(),
    )
    pp = per_patient_max(test, scores)
    py, ps, pg = (
        pp.get_column(c).to_numpy() for c in ["ever_septic", "s", "Patient_ID"]
    )
    return Result(
        model=name,
        features=feats,
        hourly_auroc=float(roc_auc_score(y, scores)),
        hourly_auroc_ci=patient_bootstrap(y, scores, g, roc_auc_score, n_boot),
        hourly_auprc=float(average_precision_score(y, scores)),
        hourly_auprc_ci=patient_bootstrap(
            y, scores, g, average_precision_score, n_boot
        ),
        patient_auroc=float(roc_auc_score(py, ps)),
        patient_auroc_ci=patient_bootstrap(py, ps, pg, roc_auc_score, n_boot),
        train_seconds=secs,
        rounds=rounds,
    )


def threshold_for_specificity(
    df: pl.DataFrame, scores: np.ndarray, specificity: float
) -> float:
    """Score threshold at which the given share of never-septic patients never alert."""
    pp = per_patient_max(df, scores)
    return float(
        np.quantile(
            pp.filter(pl.col("ever_septic") == 0).get_column("s").to_numpy(),
            specificity,
        )
    )


def alerting(
    df: pl.DataFrame,
    scores: np.ndarray,
    threshold: float,
    allowed: np.ndarray | None = None,
) -> dict[str, float]:
    """Patient-level alerting: an alert is the first hour with score >= threshold (and not blocked).

    Onset = first labelled hour + 6 (the challenge labels start 6 h early). Lead time = onset - first alert.
    """
    alert = scores >= threshold
    if allowed is not None:
        alert &= allowed
    per = (
        df.select("Patient_ID", "Hour", "SepsisLabel", "ever_septic")
        .with_columns(pl.Series("alert", alert))
        .group_by("Patient_ID")
        .agg(
            pl.col("ever_septic").first(),
            pl.col("Hour").filter(pl.col("alert")).min().alias("first_alert"),
            (pl.col("Hour").filter(pl.col("SepsisLabel") == 1).min() + 6).alias(
                "onset"
            ),
        )
    )
    septic, clean = (
        per.filter(pl.col("ever_septic") == 1),
        per.filter(pl.col("ever_septic") == 0),
    )
    early = septic.filter(pl.col("first_alert") <= pl.col("onset"))
    leads = (early["onset"] - early["first_alert"]).to_numpy()
    n_alerted = per.filter(pl.col("first_alert").is_not_null())
    return {
        "threshold": threshold,
        "patient_sensitivity_any_time": float(
            septic["first_alert"].is_not_null().mean()
        ),
        "patient_sensitivity_before_onset": float(early.height / septic.height),
        "patient_specificity": float(clean["first_alert"].is_null().mean()),
        "patient_ppv": float(n_alerted["ever_septic"].mean())
        if n_alerted.height
        else float("nan"),
        "median_lead_hours_when_early": float(np.median(leads))
        if len(leads)
        else float("nan"),
        "lead_hours_iqr_when_early": [
            float(np.percentile(leads, 25)),
            float(np.percentile(leads, 75)),
        ]
        if len(leads)
        else [],
    }


# --- guardrails -------------------------------------------------------------------------------


def ethos_allowed(df: pl.DataFrame, mode: str) -> np.ndarray:
    """ETHOS-001 (MAP and HR present) and ETHOS-002 (<= 50% of vitals+labs missing).

    mode="last_known": the patient state is the most recent value in the stay.
    mode="this_hour": only values measured in this hour count as present.
    """
    present = (
        (lambda c: pl.col(c).is_not_null())
        if mode == "last_known"
        else (lambda c: pl.col(f"{c}__obs"))
    )
    missing_frac = pl.sum_horizontal(
        (~present(c)).cast(pl.Float64) for c in MEASURED
    ) / len(MEASURED)
    return (
        df.select(present("MAP") & present("HR") & (missing_frac <= 0.5))
        .to_series()
        .to_numpy()
    )


def guardrail_report(
    test: pl.DataFrame, scores: np.ndarray, threshold: float
) -> dict[str, dict]:
    y = test.get_column("SepsisLabel").to_numpy()
    auc = lambda m: (
        float(roc_auc_score(y[m], scores[m]))
        if m.any() and y[m].min() != y[m].max()
        else None
    )
    out = {}
    for mode in ["last_known", "this_hour"]:
        allowed = ethos_allowed(test, mode)
        out[mode] = {
            "hours_allowed": float(allowed.mean()),
            "septic_label_hours_allowed": float(allowed[y == 1].mean()),
            "hourly_auroc_allowed": auc(allowed),
            "hourly_auroc_blocked": auc(~allowed),
            "alerting_with_guardrails": alerting(test, scores, threshold, allowed),
        }
    return out


# --- explanations -----------------------------------------------------------------------------


def rapport(
    row: pl.DataFrame,
    booster: lgb.Booster,
    feats: list[str],
    score: float,
    threshold: float,
    allowed: bool,
) -> str:
    contrib = booster.predict(
        row.select(feats).to_numpy(),
        pred_contrib=True,
        num_iteration=booster.best_iteration,
    )[0]
    top = sorted(
        zip(feats, contrib[:-1], row.select(feats).row(0)), key=lambda t: -abs(t[1])
    )[:5]
    r = row.row(0, named=True)
    missing = [c for c in ["MAP", "HR"] if r[c] is None]
    lines = [
        f"### Patient {r['Patient_ID']}, hour {int(r['Hour'])} (septic stay: {'yes' if r['ever_septic'] else 'no'}; label this hour: {r['SepsisLabel']})",
        "",
        f"- Risk score: {score:.3f} (alert threshold {threshold:.3f}) -> {'ALERT' if score >= threshold else 'no alert'}",
        f"- Guardrails: {'allowed' if allowed else 'BLOCKED'}"
        + (
            ""
            if allowed
            else (
                f" (ETHOS-001: missing {', '.join(missing)})"
                if missing
                else " (ETHOS-002: more than 50% of vitals and labs missing)"
            )
        ),
        "- Largest contributions (log-odds, SHAP):",
        *[
            f"  - {f} = {'missing' if v is None else f'{v:.2f}'}: {c:+.3f}"
            for f, c, v in top
        ],
        "",
    ]
    return "\n".join(lines)


# --- main -------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, default=Path("data/split"))
    ap.add_argument(
        "--discovery", type=Path, default=Path("results/causal_discovery.json")
    )
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    args = ap.parse_args()

    train, test = (
        pl.read_parquet(args.split / f"{s}.parquet") for s in ["train", "test"]
    )
    discovery = json.loads(args.discovery.read_text())
    causal = [f for f, _ in discovery["mrmr_physiological"]]
    feature_sets = {
        "causal_drivers": causal,
        "mrmr_with_time_admin": [f for f, _ in discovery["mrmr_with_time_admin"]],
        "all_features": ALL_FEATURES,
        "all_minus_time_admin": [c for c in ALL_FEATURES if c not in TIME_ADMIN],
    }
    fit, val = split_validation(train)
    print(
        f"train {train['Patient_ID'].n_unique():,} patients (hospital A), test {test['Patient_ID'].n_unique():,} (hospital B), device={args.device}"
    )

    results = [
        evaluate(
            "iculos_only",
            ["ICULOS"],
            test,
            test["ICULOS"].to_numpy().astype(float),
            0.0,
            0,
            args.n_boot,
        ),
        evaluate(
            "clinical_score",
            ["HR", "Temp", "Resp", "PaCO2", "WBC", "SBP"],
            test,
            clinical_score(test),
            0.0,
            0,
            args.n_boot,
        ),
    ]
    boosters, test_scores = {}, {}
    for name, feats in feature_sets.items():
        booster, secs = fit_lgb(fit, val, feats, args.device)
        scores = booster.predict(
            test.select(feats).to_numpy(), num_iteration=booster.best_iteration
        )
        boosters[name], test_scores[name] = booster, scores
        results.append(
            evaluate(
                name, feats, test, scores, secs, booster.best_iteration, args.n_boot
            )
        )
        print(
            f"{name:22s} hourly AUROC {results[-1].hourly_auroc:.3f}  patient AUROC {results[-1].patient_auroc:.3f}  ({secs:.1f}s, {booster.best_iteration} rounds)"
        )

    # Alerting at a fixed patient specificity on hospital A validation patients.
    alert_report = {}
    for name in ["causal_drivers", "all_features", "all_minus_time_admin"]:
        b, feats = boosters[name], feature_sets[name]
        thr = threshold_for_specificity(
            val,
            b.predict(val.select(feats).to_numpy(), num_iteration=b.best_iteration),
            TARGET_PATIENT_SPECIFICITY,
        )
        alert_report[name] = alerting(test, test_scores[name], thr)
    clin_thr = threshold_for_specificity(
        val, clinical_score(val), TARGET_PATIENT_SPECIFICITY
    )
    alert_report["clinical_score"] = alerting(test, clinical_score(test), clin_thr)

    main_scores, main_thr = (
        test_scores["causal_drivers"],
        alert_report["causal_drivers"]["threshold"],
    )
    guard = guardrail_report(test, main_scores, main_thr)

    # Example diagnostic reports: a true alert, a false alarm, a blocked hour.
    allowed = ethos_allowed(test, "last_known")
    y, sept = test["SepsisLabel"].to_numpy(), test["ever_septic"].to_numpy()
    picks = {
        "True alert (septic stay, labelled hour)": np.flatnonzero(
            (main_scores >= main_thr) & (y == 1) & allowed
        ),
        "False alarm (never-septic stay)": np.flatnonzero(
            (main_scores >= main_thr) & (sept == 0) & allowed
        ),
        "Blocked by guardrails": np.flatnonzero(~allowed & (main_scores >= main_thr)),
    }
    rng = np.random.default_rng(0)
    reports = ["# Example diagnostic reports (hospital B, causal-driver detector)", ""]
    for title, idx in picks.items():
        if len(idx):
            i = int(rng.choice(idx))
            reports += [
                f"## {title}",
                "",
                rapport(
                    test[i],
                    boosters["causal_drivers"],
                    causal,
                    float(main_scores[i]),
                    main_thr,
                    bool(allowed[i]),
                ),
            ]

    report = {
        "split": "hospital A (train) -> hospital B (test)",
        "device": args.device,
        "train_patients": train["Patient_ID"].n_unique(),
        "test_patients": test["Patient_ID"].n_unique(),
        "test_hours": test.height,
        "test_hourly_prevalence": float(test["SepsisLabel"].mean()),
        "test_patient_prevalence": float(
            test.group_by("Patient_ID")
            .agg(pl.col("ever_septic").first())["ever_septic"]
            .mean()
        ),
        "target_patient_specificity_on_validation": TARGET_PATIENT_SPECIFICITY,
        "results": [asdict(r) for r in results],
        "alerting": alert_report,
        "guardrails_causal_drivers": guard,
        "lightgbm": {"version": lgb.__version__, **LGB_PARAMS},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "detection_results.json").write_text(json.dumps(report, indent=2))
    (args.out / "example_reports.md").write_text("\n".join(reports))
    print(
        json.dumps(
            {k: report[k] for k in ["alerting", "guardrails_causal_drivers"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
