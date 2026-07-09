"""Repository-clustered bootstrap for VibeTest ex0 selective macro F1.

WIP / DOES NOT YET RECONCILE: the per-case files on the qwen36-prompt-examples
branch reproduce a base non-inconclusive rate of 0.86 and peak F1 ~0.60, not the
paper's 0.77 / 0.802 for ex0 -- so this branch's `_case-score.jsonl` is not the
exact ex0 artifact behind Table 1. The clustered CI is therefore NOT consistent
with the published numbers and must not be cited until the correct per-case
artifact (matching 0.802 @ 0.56 coverage) and scorer are located. The case-level
CIs in bootstrap_cis.py DO match the published point estimates and are the ones
used in the paper.


Reads per-(repo,property) Qwen3.6 outputs from the qwen36-prompt-examples git
branch, reproduces the selective curve, and if the peak matches the reported
0.802 @ ~0.56 coverage, resamples REPOSITORIES (not individual cases) to get a
CI that respects within-repository correlation.
"""
import subprocess, json, re, sys
import numpy as np

BRANCH = "qwen36-prompt-examples"
FILES = [f"results/synthetic/synthetic_kaggle_{d}_AT-Qwen-Qwen3.6-35B-A3B-FP8_case-score.jsonl"
         for d in ("titanic", "nlp", "diabetic")]
REPO = r"C:\Users\User\Downloads\vibetest-ml\vibetest"


def load():
    cases = []  # (repo_uid, score, is_inconclusive, gt_fail)
    for f in FILES:
        raw = subprocess.run(["git", "show", f"{BRANCH}:{f}"], cwd=REPO,
                             capture_output=True, text=True, encoding="utf-8").stdout
        for line in raw.splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            repo_uid = f"{d.get('dataset')}/{d.get('repo_name')}"
            gt = {g["property_id"]: g["label"] for g in d["ground_truth_by_property"]}
            for t in d["tests"]:
                pid = t.get("metadata", {}).get("property_id")
                if pid not in gt:
                    continue
                desc = t.get("description", "")
                mv = re.search(r"VERDICT:\s*([A-Z]+)", desc)
                ms = re.search(r"CASE_SCORE:\s*([0-9.]+)", desc)
                if not mv:
                    continue
                verdict = mv.group(1)
                score = float(ms.group(1)) if ms else 0.5
                incon = verdict.startswith("INCON")
                cases.append((repo_uid, score, incon, int(gt[pid] == 1)))
    return cases


def macro_f1_covered(cases, t):
    # covered = non-inconclusive AND (score>=t or score<=1-t); pred fail if score>=t
    ff = fp = pf = pp = 0
    for _, s, incon, gtf in cases:
        if incon:
            continue
        if s >= t:
            pred = 1
        elif s <= 1 - t:
            pred = 0
        else:
            continue
        if gtf and pred: ff += 1
        elif (not gtf) and pred: fp += 1
        elif gtf and not pred: pf += 1
        else: pp += 1
    n = ff + fp + pf + pp
    fr = ff / (ff + pf) if ff + pf else 0; fpr = ff / (ff + fp) if ff + fp else 0
    ff1 = 2 * fr * fpr / (fr + fpr) if fr + fpr else 0
    pr = pp / (pp + fp) if pp + fp else 0; ppr = pp / (pp + pf) if pp + pf else 0
    pf1 = 2 * pr * ppr / (pr + ppr) if pr + ppr else 0
    return 0.5 * (ff1 + pf1), n


cases = load()
total = len(cases)
print(f"loaded {total} (repo,property) cases; base non-incon rate="
      f"{sum(1 for c in cases if not c[2])/total:.3f}")

# reproduce selective curve, find peak
best = (0, 0, 0)
for t in np.arange(0.50, 1.001, 0.01):
    f1, n = macro_f1_covered(cases, t)
    if f1 > best[0]:
        best = (f1, t, n / total)
print(f"reproduced peak macro F1={best[0]:.4f} at t={best[1]:.2f}, coverage={best[2]:.3f}")
print("  (paper reports 0.802 @ coverage 0.56)")

peak_t = best[1]
# cluster bootstrap: resample repositories with replacement
repos = {}
for c in cases:
    repos.setdefault(c[0], []).append(c)
repo_ids = list(repos.keys())
rng = np.random.default_rng(7)
B = 5000
vals = []
for _ in range(B):
    pick = rng.choice(len(repo_ids), size=len(repo_ids), replace=True)
    boot = [c for i in pick for c in repos[repo_ids[i]]]
    f1, _ = macro_f1_covered(boot, peak_t)
    vals.append(f1)
lo, hi = np.percentile(vals, [2.5, 97.5])
print(f"\nex0 REPO-CLUSTERED 95% CI at peak: [{lo:.3f}, {hi:.3f}]  ({len(repo_ids)} repos)")
print("compare case-level CI (from bootstrap_cis.py): [0.769, 0.834]")
