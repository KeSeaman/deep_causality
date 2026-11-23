"""Reference SURD (Martinez-Sanchez et al. 2024), transcribed with explicit keepdims broadcasting.

Cross-checks the Rust port: see backend/src/causality/surd_aggregate.rs::dump_for_reference.
The LIB comparison runs only if the dump contains LIB lines.
"""
import ast, re, sys
from itertools import combinations
import numpy as np

def mylog(x):
    out = np.zeros_like(x); m = x > 0; out[m] = np.log2(x[m]); return out

def surd(p):
    p = p / p.sum(); ntot = p.ndim; nvars = ntot - 1; nt = p.shape[0]; inds = tuple(range(1, ntot))
    p_s = p.sum(axis=inds, keepdims=True)
    combs, Is = [], {}
    for i in inds:
        for j in combinations(inds, i):
            combs.append(j); noj = tuple(sorted(set(inds) - set(j)))
            p_a = p.sum(axis=(0, *noj), keepdims=True); p_as = p.sum(axis=noj, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                p_a_s = np.nan_to_num(p_as / p_s); p_s_a = np.nan_to_num(p_as / p_a)
            Is[j] = (p_a_s * (mylog(p_s_a) - mylog(p_s))).sum(axis=j).ravel()
    I_R, I_S = {}, {}
    for t in range(nt):
        I1 = np.array([Is[c][t] for c in combs]); i1 = np.argsort(I1, kind="stable")
        lab = [combs[k] for k in i1]; lens = np.array([len(l) for l in lab]); I1 = I1[i1]
        for l in range(1, lens.max()):
            m = I1[lens == l].max(); idx = np.where((lens == l + 1) & (I1 < m))[0]; I1[idx] = 0
        i1 = np.argsort(I1, kind="stable"); lab = [lab[k] for k in i1]; Di = np.diff(I1[i1], prepend=0.0)
        red = list(inds)
        for d, ll in zip(Di, lab):
            info = d * p_s.ravel()[t]
            if len(ll) == 1: I_R[tuple(red)] = I_R.get(tuple(red), 0) + info; red.remove(ll[0])
            else: I_S[ll] = I_S.get(ll, 0) + info
    return I_R, I_S

def parse(s):
    return {tuple(ast.literal_eval(k)): float(v) for k, v in (kv.rsplit(":", 1) for kv in s.split(";") if kv)}

lines = open(sys.argv[1]).read().splitlines()
for i, line in enumerate(lines):
    if not line.startswith("CASE"): continue
    shape = ast.literal_eval(re.search(r"shape=(\[.*?\])", line).group(1)); p = np.array(ast.literal_eval(re.search(r"p=(\[.*\])", line).group(1))).reshape(shape)
    R, Sy = surd(p)
    port = re.match(r"\s+PORT R=(.*) S=(.*)", lines[i + 1]); pr, ps = parse(port.group(1)), parse(port.group(2))
    dev_port = max(abs(R.get(k, 0) - pr.get(k, 0)) for k in set(R) | set(pr)) if (R or pr) else 0
    dev_port = max(dev_port, max((abs(Sy.get(k, 0) - ps.get(k, 0)) for k in set(Sy) | set(ps)), default=0))
    libline = lines[i + 2] if i + 2 < len(lines) else ""
    if not libline.strip().startswith("LIB"): dev_lib = "not dumped"
    elif "error" in libline: dev_lib = "library error"
    else:
        lm = re.match(r"\s+LIB  R=(.*) S=(.*)", libline); lr, ls = parse(lm.group(1)), parse(lm.group(2))
        dev_lib = max(max(abs(R.get(k, 0) - lr.get(k, 0)) for k in set(R) | set(lr)), max((abs(Sy.get(k, 0) - ls.get(k, 0)) for k in set(Sy) | set(ls)), default=0))
        dev_lib = f"{dev_lib:.2e}"
    print(f"shape={shape}: max |port - reference| = {dev_port:.2e}; max |library - reference| = {dev_lib}")
