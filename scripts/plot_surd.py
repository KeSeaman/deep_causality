"""Graphviz diagram of the SURD decomposition in results/causal_discovery.json.

Sources (left) connect to information terms (middle): unique (one source), redundant (shared by
a set) and synergistic (only in combination). Each term connects to the label (right) with width
proportional to its bits. Lines are undirected: SURD measures information structure, not
causal direction.

    /usr/bin/python3 scripts/plot_surd.py --discovery results/causal_discovery.json --out results/surd_decomposition
"""

import argparse
import json
import subprocess
from pathlib import Path

KIND_COLOR = {"Unique": "#2a78d6", "Redundant": "#eb6834", "Synergistic": "#1baf7a"}
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
MIN_SYNERGY_BITS = 1.5e-4  # smaller synergistic terms are folded into one node


def node_id(kind: str, features: list[str]) -> str:
    return f"{kind[0]}_{'_'.join(features)}"


def to_dot(surd: dict) -> str:
    by_bits = lambda ts: sorted(ts, key=lambda t: -t["bits"])
    terms = [("Unique", t) for t in by_bits(surd["unique"])] + [
        ("Redundant", t) for t in by_bits(surd["redundant"])
    ]
    syn = sorted(surd["synergistic"], key=lambda t: -t["bits"])
    big = [t for t in syn if t["bits"] >= MIN_SYNERGY_BITS]
    rest = [t for t in syn if t["bits"] < MIN_SYNERGY_BITS]
    terms += [("Synergistic", t) for t in big]
    max_bits = max(t["bits"] for _, t in terms)
    width = lambda b: 1.0 + 5.0 * b / max_bits

    pct = 100 * surd["joint_mutual_info_bits"] / surd["target_entropy_bits"]
    title = (
        f"SURD on hospital A: what five vital/lab drivers tell us about the 6-hour sepsis label\\n"
        f"Jointly {surd['joint_mutual_info_bits']:.4f} of {surd['target_entropy_bits']:.4f} bits ({pct:.0f}%); "
        f"unique {surd['total_unique_bits']:.4f} · redundant {surd['total_redundant_bits']:.4f} · "
        f"synergistic {surd['total_synergistic_bits']:.4f} bits"
    )
    lines = [
        "graph SURD {",
        f'  graph [rankdir=LR, bgcolor="{SURFACE}", fontname="Helvetica", fontsize=13, fontcolor="{INK}", '
        f'label="{title}", labelloc=t, nodesep=0.12, ranksep=1.1, pad=0.3, ordering=out];',
        f'  node [fontname="Helvetica", fontsize=10, fontcolor="{INK}", style="filled,rounded", fillcolor="white", shape=box, penwidth=1.5];',
        f'  edge [color="#c3c2b7"];',
        f'  label [label="Sepsis within 6 h\\n(H = {surd["target_entropy_bits"]:.3f} bits)", shape=ellipse, color="{INK}", penwidth=2];',
    ]
    lines += [f'  "{s}" [label="{s}", color="{MUTED}"];' for s in surd["sources"]]

    for kind, t in terms:
        nid, color = node_id(kind, t["features"]), KIND_COLOR[kind]
        members = (
            ", ".join(t["features"])
            if len(t["features"]) < len(surd["sources"])
            else "all five"
        )
        lines.append(
            f'  "{nid}" [label="{kind}\\n{members}\\n{t["bits"]:.1e} bits", color="{color}", fontsize=9];'
        )
        lines += [
            f'  "{f}" -- "{nid}" [color="#d6d5cf", penwidth=0.8];'
            for f in t["features"]
        ]
        lines.append(
            f'  "{nid}" -- label [color="{color}", penwidth={width(t["bits"]):.2f}];'
        )

    if rest:
        bits = sum(t["bits"] for t in rest)
        color = KIND_COLOR["Synergistic"]
        lines.append(
            f'  other [label="Synergistic\\n{len(rest)} smaller sets\\n{bits:.1e} bits in total", color="{color}", fontsize=9, style="filled,rounded,dashed"];'
        )
        lines.append(
            f'  other -- label [color="{color}", penwidth={width(bits):.2f}, style=dashed];'
        )

    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--discovery", type=Path, default=Path("results/causal_discovery.json")
    )
    ap.add_argument("--out", type=Path, default=Path("results/surd_decomposition"))
    args = ap.parse_args()

    dot = to_dot(json.loads(args.discovery.read_text())["surd"])
    args.out.with_suffix(".dot").write_text(dot)
    for fmt in ["svg", "png"]:
        subprocess.run(
            [
                "dot",
                f"-T{fmt}",
                "-Gdpi=110",
                str(args.out.with_suffix(".dot")),
                "-o",
                str(args.out.with_suffix(f".{fmt}")),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
