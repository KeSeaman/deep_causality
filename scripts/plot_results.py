"""README figure: hourly AUROC on hospital B with patient-bootstrap 95% CIs.

    /usr/bin/python3 scripts/plot_results.py --results results/detection_results.json --out results/detection_auroc.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

LABELS = {
    "iculos_only": "ICU length of stay only",
    "clinical_score": "SIRS + qSOFA score",
    "causal_drivers": "Causal drivers (this project)",
    "mrmr_with_time_admin": "mRMR incl. time/admin fields",
    "all_minus_time_admin": "All features, no time/admin",
    "all_features": "All features",
}
HIGHLIGHT, CONTEXT, INK, MUTED, SURFACE = "#2a78d6", "#52514e", "#0b0b0b", "#52514e", "#fcfcfb"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/detection_results.json"))
    ap.add_argument("--out", type=Path, default=Path("results/detection_auroc.png"))
    args = ap.parse_args()

    res = sorted(json.loads(args.results.read_text())["results"], key=lambda r: r["hourly_auroc"])
    fig, ax = plt.subplots(figsize=(7.5, 3.4), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, r in enumerate(res):
        color = HIGHLIGHT if r["model"] == "causal_drivers" else CONTEXT
        lo, hi = r["hourly_auroc_ci"]
        ax.plot([lo, hi], [i, i], color=color, lw=2, solid_capstyle="round")
        ax.plot(r["hourly_auroc"], i, "o", ms=8, color=color, mec=SURFACE, mew=2)
        ax.text(hi + 0.006, i, f"{r['hourly_auroc']:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(res)), [LABELS[r["model"]] for r in res], fontsize=9, color=INK)
    ax.set_xlim(0.55, 0.82)
    ax.set_xlabel("Hourly AUROC on hospital B (external), patient-bootstrap 95% CI", fontsize=9, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="x", color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, facecolor=SURFACE)


if __name__ == "__main__":
    main()
