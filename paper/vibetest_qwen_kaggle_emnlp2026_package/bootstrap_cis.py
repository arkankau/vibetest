"""Bootstrap / Wilson confidence intervals for the paper's headline numbers.

No model re-runs needed: we resample the existing per-case outcomes.

- Synthetic selective macro F1 (VibeTest ex0/ex10/ex20, reviewer modes, Codex):
  reconstruct the covered case-level 2x2 outcome from the covered-confusion
  counts at each method's reported (peak selective-F1) operating point, then
  bootstrap macro F1 over covered cases.
- Audit proportions (operationalization mismatch, model miss out of 53) and the
  auditable real fail-precision (122/152): Wilson score intervals.
"""
import numpy as np
import pandas as pd
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "supporting_docs", "paper_synthetic_selective_f1_coverage.csv")
B = 20000
rng = np.random.default_rng(12345)


def macro_f1_from_cells(ff, fp, pf, pp):
    # ff=true-fail/pred-fail, fp=true-pass/pred-fail, pf=true-fail/pred-pass, pp=true-pass/pred-pass
    fail_p = ff / (ff + fp) if (ff + fp) else 0.0
    fail_r = ff / (ff + pf) if (ff + pf) else 0.0
    fail_f1 = 2 * fail_p * fail_r / (fail_p + fail_r) if (fail_p + fail_r) else 0.0
    pass_p = pp / (pp + pf) if (pp + pf) else 0.0
    pass_r = pp / (pp + fp) if (pp + fp) else 0.0
    pass_f1 = 2 * pass_p * pass_r / (pass_p + pass_r) if (pass_p + pass_r) else 0.0
    return 0.5 * (fail_f1 + pass_f1)


def bootstrap_macro_f1(ff, fp, pf, pp):
    cells = np.array([ff, fp, pf, pp], dtype=int)
    n = cells.sum()
    labels = np.repeat(np.arange(4), cells)  # 0=ff,1=fp,2=pf,3=pp
    point = macro_f1_from_cells(ff, fp, pf, pp)
    vals = np.empty(B)
    for b in range(B):
        s = rng.choice(labels, size=n, replace=True)
        c = np.bincount(s, minlength=4)
        vals[b] = macro_f1_from_cells(*c)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, lo, hi, n


def wilson(k, n, z=1.96):
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, center - half, center + half


df = pd.read_csv(CSV)
print("=== Synthetic selective macro F1 (95% bootstrap CI, peak operating point) ===")
for series, g in df.groupby("series", sort=False):
    if "TrainCheck" in series:
        continue
    row = g.loc[g["covered_macro_f1"].idxmax()]
    ff, fp, pf, pp = (int(row.covered_fail_tp), int(row.covered_fail_fp),
                      int(row.covered_fail_fn), int(row.covered_pass_tp))
    pt, lo, hi, n = bootstrap_macro_f1(ff, fp, pf, pp)
    name = series.split(" (")[0]
    print(f"  {name:28s} F1={pt:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (n_covered={n})")

# Codex full-run: covered confusion TP=33 FP=27 FN=0 TN=2
pt, lo, hi, n = bootstrap_macro_f1(33, 27, 0, 2)
print(f"  {'Codex reviewer (full)':28s} F1={pt:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (n_covered={n})")

print("\n=== Audit proportions (95% Wilson CI) ===")
for name, k, n in [("Operationalization mismatch", 24, 53),
                   ("Model miss", 29, 53)]:
    p, lo, hi = wilson(k, n)
    print(f"  {name:28s} {k}/{n} = {p:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

print("\n=== Real auditable fail-precision (95% Wilson CI) ===")
p, lo, hi = wilson(122, 152)
print(f"  {'TRUE_FAIL / sampled fails':28s} 122/152 = {p:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
