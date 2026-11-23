import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import polars as pl

    path = Path(__file__).resolve().parents[1] / "data" / "all" / "dataset.parquet"
    df = pl.read_parquet(path)
    return df, mo, pl


@app.cell
def _(df, mo, pl):
    patients = df.group_by("Patient_ID").agg(pl.col("SepsisLabel").max())
    mo.md(
        f"**{df.height:,} patient-hours, {patients.height:,} patients.** "
        f"Hourly label prevalence {df['SepsisLabel'].mean():.2%}; septic stays {patients['SepsisLabel'].mean():.2%}. "
        f"Hospital A: {df.filter(pl.col('Patient_ID') < 100_000)['Patient_ID'].n_unique():,} patients; "
        f"hospital B: {df.filter(pl.col('Patient_ID') >= 100_000)['Patient_ID'].n_unique():,}."
    )
    return


@app.cell
def _(df, pl):
    # Share of hours with each variable measured: labs are sparse, which is why the
    # guardrails and the "not yet measured" SURD state matter.
    (
        df.drop("Patient_ID", "SepsisLabel", "Hour")
        .select(pl.all().is_not_null().mean())
        .unpivot(variable_name="variable", value_name="share_measured")
        .sort("share_measured", descending=True)
    )
    return


if __name__ == "__main__":
    app.run()
