"""Prepare the PhysioNet 2019 data for causal discovery and detection.

- Hospital A (Patient_ID < 100000) -> data/split/train.parquet; hospital B -> data/split/test.parquet.
  Training on one hospital and testing on the other is an external validation.
- Within each ICU stay, values are carried forward (last known value), so every feature at hour t
  uses only information available at hour t.
- ``<col>__obs`` flags record whether the value was measured in that hour (used by the guardrails).

    /usr/bin/python3 scripts/prepare_data.py --data data/all/dataset.parquet --out data/split
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

VITALS = [
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
]  # EtCO2 dropped: string column, ~all missing
LABS = [
    "BaseExcess",
    "HCO3",
    "FiO2",
    "pH",
    "PaCO2",
    "SaO2",
    "AST",
    "BUN",
    "Alkalinephos",
    "Calcium",
    "Chloride",
    "Creatinine",
    "Bilirubin_direct",
    "Glucose",
    "Lactate",
    "Magnesium",
    "Phosphate",
    "Potassium",
    "Bilirubin_total",
    "TroponinI",
    "Hct",
    "Hgb",
    "PTT",
    "WBC",
    "Fibrinogen",
    "Platelets",
]
MEASURED = [*VITALS, *LABS]


def prepare(path: Path) -> pl.DataFrame:
    return (
        pl.scan_parquet(path)
        .drop("EtCO2")
        .with_columns(
            pl.col("Patient_ID").cast(pl.Int64), pl.col("SepsisLabel").cast(pl.Int8)
        )
        .sort("Patient_ID", "Hour")
        .with_columns(pl.col(c).is_not_null().alias(f"{c}__obs") for c in MEASURED)
        .with_columns(pl.col(c).forward_fill().over("Patient_ID") for c in MEASURED)
        .with_columns(
            pl.col("SepsisLabel").max().over("Patient_ID").alias("ever_septic")
        )
        .collect()
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/all/dataset.parquet"))
    ap.add_argument("--out", type=Path, default=Path("data/split"))
    args = ap.parse_args()

    df = prepare(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, part in [
        ("train", df.filter(pl.col("Patient_ID") < 100_000)),
        ("test", df.filter(pl.col("Patient_ID") >= 100_000)),
    ]:
        part.write_parquet(args.out / f"{name}.parquet")
        print(
            f"{name}: {part.height:,} hours, {part['Patient_ID'].n_unique():,} patients, "
            f"{part.group_by('Patient_ID').agg(pl.col('ever_septic').first())['ever_septic'].mean():.2%} septic"
        )


if __name__ == "__main__":
    main()
