"""Model x dataset native-operating-point comparison + token cost.

Both models are evaluated at their NATIVE operating point (covered =
non-inconclusive verdicts; no case_score thresholding), because GPT-5-mini
outputs carry no case_score. This is a cross-model robustness check, distinct
from the selective (peak) Qwen numbers in the headline table.
"""
import subprocess, json, re, os
import numpy as np

REPO = r"C:\Users\User\Downloads\vibetest-ml\vibetest"
# Canonical headline run: the ex0 "bands" files on ml-vibetest-a reproduce the
# paper's 0.773 base coverage exactly. (An earlier version used a different FP8
# run that over-counted tokens ~2x; these are the correct headline numbers.)
BRANCH = "ml-vibetest-a"
DS = ["titanic", "nlp", "diabetic"]


def read_lines(path, from_branch):
    if from_branch:
        return subprocess.run(["git", "show", f"{BRANCH}:{path}"], cwd=REPO,
                              capture_output=True, text=True, encoding="utf-8").stdout.splitlines()
    with open(os.path.join(REPO, path), encoding="utf-8") as fh:
        return fh.read().splitlines()


def score(lines):
    ff = fp = pf = pp = 0
    out_tok = 0
    n = 0
    for line in lines:
        if not line.strip():
            continue
        d = json.loads(line)
        gt = {g["property_id"]: g["label"] for g in d["ground_truth_by_property"]}
        u = d.get("usage", {}).get("usage_totals", {})
        out_tok += u.get("output_tokens", 0)
        for t in d["tests"]:
            pid = t.get("metadata", {}).get("property_id")
            if pid not in gt:
                continue
            n += 1
            mv = re.search(r"VERDICT:\s*([A-Z]+)", t.get("description", ""))
            if not mv:
                continue
            v = mv.group(1)
            if v.startswith("INCON"):
                continue
            pred = 1 if v == "FAIL" else 0
            gtf = int(gt[pid] == 1)
            if gtf and pred: ff += 1
            elif (not gtf) and pred: fp += 1
            elif gtf and not pred: pf += 1
            else: pp += 1
    cov_n = ff + fp + pf + pp
    fr = ff / (ff + pf) if ff + pf else 0; fpr = ff / (ff + fp) if ff + fp else 0
    ff1 = 2 * fr * fpr / (fr + fpr) if fr + fpr else 0
    pr = pp / (pp + fp) if pp + fp else 0; ppr = pp / (pp + pf) if pp + pf else 0
    pf1 = 2 * pr * ppr / (pr + ppr) if pr + ppr else 0
    macro = 0.5 * (ff1 + pf1)
    return dict(n=n, cov=cov_n / n if n else 0, macro=macro,
                out_tok=out_tok, tok_per=out_tok / n if n else 0)


MODELS = [
    ("Qwen3.6", lambda d: (f"results/synthetic/synthetic_kaggle_{d}_AT-Qwen-Qwen3.6-35B-A3B-FP8-static_examples0_case_score_bands.jsonl", True)),
    ("GPT-5-mini", lambda d: (f"results/synthetic/synthetic_kaggle_{d}_AT-gpt-5-mini.jsonl", False)),
]

print(f"{'Model':<11}{'Dataset':<10}{'Cov':>7}{'MacroF1':>9}{'OutTok/prop':>13}")
agg = {}
for mname, pathfn in MODELS:
    tot = dict(ff_n=0, cov_num=0, tok=0, n=0)
    rows = []
    for d in DS:
        path, br = pathfn(d)
        r = score(read_lines(path, br))
        rows.append((d, r))
        print(f"{mname:<11}{d:<10}{r['cov']:>7.3f}{r['macro']:>9.3f}{r['tok_per']:>13.0f}")
    # aggregate macro across datasets (unweighted mean of per-dataset macro)
    macros = [r['macro'] for _, r in rows]
    covs = [r['cov'] for _, r in rows]
    toks = [r['tok_per'] for _, r in rows]
    print(f"{mname:<11}{'MEAN':<10}{np.mean(covs):>7.3f}{np.mean(macros):>9.3f}{np.mean(toks):>13.0f}\n")
