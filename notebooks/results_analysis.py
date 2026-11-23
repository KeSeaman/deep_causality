import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt

    root = Path(__file__).resolve().parents[1] / "results"
    discovery = json.loads((root / "causal_discovery.json").read_text())
    detection = json.loads((root / "detection_results.json").read_text())
    return detection, discovery, mo, plt, root


@app.cell
def _(detection, mo):
    rows = [
        f"| {r['model']} | {r['hourly_auroc']:.3f} ({r['hourly_auroc_ci'][0]:.3f}-{r['hourly_auroc_ci'][1]:.3f}) "
        f"| {r['hourly_auprc']:.3f} | {r['patient_auroc']:.3f} |"
        for r in detection["results"]
    ]
    mo.md(
        f"## Detection on hospital B ({detection['test_patients']:,} patients)\n\n"
        "| Detector | Hourly AUROC (95% CI) | Hourly AUPRC | Patient AUROC |\n| --- | --- | --- | --- |\n"
        + "\n".join(rows)
    )
    return


@app.cell
def _(detection, plt):
    res = detection["results"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    y = range(len(res))
    ax.errorbar(
        [r["hourly_auroc"] for r in res], y,
        xerr=[[r["hourly_auroc"] - r["hourly_auroc_ci"][0] for r in res], [r["hourly_auroc_ci"][1] - r["hourly_auroc"] for r in res]],
        fmt="o", capsize=3,
    )
    ax.set_yticks(list(y), [r["model"] for r in res])
    ax.axvline(0.5, ls=":", c="grey")
    ax.set_xlabel("Hourly AUROC, hospital B (patient-bootstrap 95% CI)")
    fig.tight_layout()
    fig
    return


@app.cell
def _(discovery, mo):
    s = discovery["surd"]
    terms = lambda k: "\n".join(f"| {', '.join(t['features'])} | {t['bits']:.5f} |" for t in s[k][:6])
    mo.md(
        f"## SURD on hospital A: {', '.join(s['sources'])}\n\n"
        f"Label entropy {s['target_entropy_bits']:.4f} bits; the sources jointly carry {s['joint_mutual_info_bits']:.4f} bits "
        f"(leak fraction {s['info_leak_fraction']:.3f}).\n\n"
        f"Unique {s['total_unique_bits']:.5f} · redundant {s['total_redundant_bits']:.5f} · synergistic {s['total_synergistic_bits']:.5f} bits\n\n"
        "| Synergistic set | Bits |\n| --- | --- |\n" + terms("synergistic")
        + "\n\n| Redundant set | Bits |\n| --- | --- |\n" + terms("redundant")
    )
    return


@app.cell
def _(detection, mo):
    g = detection["guardrails_causal_drivers"]
    a = detection["alerting"]
    mo.md(
        "## Alerting and guardrails\n\n"
        "| Detector | Sensitivity before onset | Specificity | PPV | Median lead (h) |\n| --- | --- | --- | --- | --- |\n"
        + "\n".join(
            f"| {k} | {v['patient_sensitivity_before_onset']:.2f} | {v['patient_specificity']:.2f} | {v['patient_ppv']:.2f} | {v['median_lead_hours_when_early']:.0f} |"
            for k, v in a.items()
        )
        + "\n\n| Guardrail reading | Hours allowed | Septic hours allowed | AUROC allowed | AUROC blocked |\n| --- | --- | --- | --- | --- |\n"
        + "\n".join(
            f"| {m} | {v['hours_allowed']:.1%} | {v['septic_label_hours_allowed']:.1%} | {v['hourly_auroc_allowed'] or float('nan'):.3f} | {v['hourly_auroc_blocked'] or float('nan'):.3f} |"
            for m, v in g.items()
        )
    )
    return


if __name__ == "__main__":
    app.run()
