"""Evidence-quality metric for VibeTest FAIL verdicts.

An independent verifier (GPT-5.4-mini, different family from the Qwen3.6 tester)
checks, for each FAIL verdict, whether the cited evidence actually supports it.
We aggregate the per-repo scoring fields across the three synthetic datasets.

Reported:
- support-rate | cited: of FAIL verdicts whose evidence was checkable, the
  fraction the verifier confirms is supported.
- support-rate | all: of ALL FAIL verdicts, the fraction with verifier-confirmed
  supporting evidence (fails without checkable evidence count against).
- raw fail precision vs evidence-verified fail precision (the "evidence tax":
  how many detections lose credit once the evidence must hold up).
"""
import subprocess, json
import numpy as np

BRANCH = "qwen36-prompt-examples"
REPO = r"C:\Users\User\Downloads\vibetest-ml\vibetest"
DS = ["titanic", "nlp", "diabetic"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


agg = dict(fail_pred=0, ev_checked=0, ev_ok=0,
           vtp=0, vfp=0, vfn=0, vtn=0, raw_tp=0, raw_fp=0)
for d in DS:
    f = f"results/synthetic/synthetic_kaggle_{d}_AT-Qwen-Qwen3.6-35B-A3B-FP8_case-score.jsonl"
    raw = subprocess.run(["git", "show", f"{BRANCH}:{f}"], cwd=REPO,
                         capture_output=True, text=True, encoding="utf-8").stdout
    for line in raw.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        s = rec["scoring"]
        agg["fail_pred"] += s.get("fail_predictions", 0)
        agg["ev_checked"] += s.get("fail_evidence_verified_count", 0)
        agg["ev_ok"] += s.get("fail_evidence_verified_correct_count", 0)
        agg["vtp"] += s.get("verified_fail_tp", 0)
        agg["vfp"] += s.get("verified_fail_fp", 0)
        agg["vfn"] += s.get("verified_fail_fn", 0)
        agg["vtn"] += s.get("verified_fail_tn", 0)
        # raw fail precision from GT labels vs FAIL verdict
        gt = {g["property_id"]: g["label"] for g in rec["ground_truth_by_property"]}
        for t in rec["tests"]:
            pid = t.get("metadata", {}).get("property_id")
            desc = t.get("description", "")
            import re
            mv = re.search(r"VERDICT:\s*([A-Z]+)", desc)
            if not mv or mv.group(1) != "FAIL" or pid not in gt:
                continue
            if gt[pid] == 1:
                agg["raw_tp"] += 1
            else:
                agg["raw_fp"] += 1

print("=== Evidence quality (independent GPT-5.4-mini verifier over Qwen3.6 FAIL verdicts) ===")
p, lo, hi = wilson(agg["ev_ok"], agg["ev_checked"])
print(f"support-rate | cited : {agg['ev_ok']}/{agg['ev_checked']} = {p:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
p2, lo2, hi2 = wilson(agg["ev_ok"], agg["fail_pred"])
print(f"support-rate | all   : {agg['ev_ok']}/{agg['fail_pred']} = {p2:.3f}  95% CI [{lo2:.3f}, {hi2:.3f}]")
print(f"  ({agg['fail_pred']-agg['ev_checked']} FAIL verdicts had no checkable cited evidence)")

raw_prec = agg["raw_tp"] / (agg["raw_tp"] + agg["raw_fp"]) if (agg["raw_tp"] + agg["raw_fp"]) else 0
ver_prec = agg["vtp"] / (agg["vtp"] + agg["vfp"]) if (agg["vtp"] + agg["vfp"]) else 0
print("\n=== Evidence tax: fail precision before vs after requiring supported evidence ===")
print(f"raw fail precision (verdict vs GT)          : {raw_prec:.3f}  ({agg['raw_tp']}/{agg['raw_tp']+agg['raw_fp']})")
print(f"evidence-verified fail precision            : {ver_prec:.3f}  ({agg['vtp']}/{agg['vtp']+agg['vfp']})")
