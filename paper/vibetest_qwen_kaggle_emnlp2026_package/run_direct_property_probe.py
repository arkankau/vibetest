"""Same-checkpoint direct-property baseline via OpenRouter Qwen3.6-35B.

Direct prompting: give the model the notebook + one property and ask PASS/FAIL/
INCONCLUSIVE + case_score, with NO evidence rubric, NO tools, NO execution.
Compared against VibeTest on the SAME 5 repos and SAME checkpoint (the first 5
repos of the OpenRouter Titanic AT run), so it is a fair same-checkpoint probe.
"""
import os, json, re, glob
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

VT = Path(r"C:\Users\User\Downloads\vibetest-ml\vibetest")
load_dotenv(VT / ".env")
client = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"])
MODEL = "qwen/qwen3.6-35b-a3b-20260415"
N_REPOS = 5
LABELS = VT / "synth-data/injected/labels_kaggle_titanic.jsonl"
VIBETEST_RUN = VT / "results/synthetic/synthetic_kaggle_titanic_AT-Qwen-openrouter.jsonl"

PROMPT = """You are checking whether a machine-learning repository satisfies a property.
You are given the repository's notebook source and one property to check.
Decide whether the repository satisfies the property, using only the source below.

Repository notebook source:
```
{src}
```

Property to check: {prop}

Respond in EXACTLY this format and nothing else:
VERDICT: <PASS or FAIL or INCONCLUSIVE>
CASE_SCORE: <number in [0,1]; 0 = clearly satisfies, 1 = clearly violates>
REASON: <one sentence>"""


def notebook_text(repo_dir):
    nbs = glob.glob(str(Path(VT) / repo_dir / "**" / "*.ipynb"), recursive=True)
    if not nbs:
        return ""
    nb = json.load(open(nbs[0], encoding="utf-8"))
    out = []
    for c in nb.get("cells", []):
        s = "".join(c.get("source", []))
        if s.strip():
            out.append(("# [markdown]\n" if c.get("cell_type") == "markdown" else "") + s)
    txt = "\n\n".join(out)
    return txt[:60000]  # keep prompt bounded


def ask(src, prop):
    r = client.chat.completions.create(
        model=MODEL, temperature=0.0, max_tokens=2000,
        messages=[{"role": "user", "content": PROMPT.format(src=src, prop=prop)}])
    t = r.choices[0].message.content or ""
    v = re.search(r"VERDICT:\s*([A-Z]+)", t)
    s = re.search(r"CASE_SCORE:\s*([0-9.]+)", t)
    return (v.group(1) if v else "INCONCLUSIVE"), (float(s.group(1)) if s else 0.5)


# load probe repos (rows 0..N-1) with GT + notebook
rows = []
for i, line in enumerate(open(LABELS, encoding="utf-8")):
    if i >= N_REPOS:
        break
    d = json.loads(line)
    rows.append(d)

# direct-property calls
jobs = []
for d in rows:
    src = notebook_text(d["output_repo_path"])
    for g in d["ground_truth_by_property"]:
        jobs.append((d["row_index"], g["property_id"], g["label"], g["property_text"], src))
print(f"direct-property: {len(jobs)} (repo,property) calls on {N_REPOS} repos via {MODEL}")


def run(job):
    ri, pid, label, prop, src = job
    v, sc = ask(src, prop)
    return dict(row=ri, pid=pid, gt=label, verdict=v, score=sc)


with ThreadPoolExecutor(max_workers=6) as ex:
    direct = list(ex.map(run, jobs))

# VibeTest on the same rows/props (same checkpoint)
vt = {}
for line in open(VIBETEST_RUN, encoding="utf-8"):
    if not line.strip():
        continue
    d = json.loads(line)
    ri = d.get("synthetic_row_index")
    if ri is None or ri >= N_REPOS:
        continue
    gt = {g["property_id"]: g["label"] for g in d["ground_truth_by_property"]}
    for t in d["tests"]:
        pid = t.get("metadata", {}).get("property_id")
        m = re.search(r"VERDICT:\s*([A-Z]+)", t.get("description", ""))
        if pid in gt and m:
            vt[(ri, pid)] = dict(row=ri, pid=pid, gt=gt[pid], verdict=m.group(1))


def metrics(rows):
    ff = fp = pf = pp = incon = 0
    for r in rows:
        v = r["verdict"]; gtf = r["gt"] == 1
        if v.startswith("INCON"):
            incon += 1; continue
        pred = v == "FAIL"
        if gtf and pred: ff += 1
        elif (not gtf) and pred: fp += 1
        elif gtf and not pred: pf += 1
        else: pp += 1
    n = len(rows); cov = (n - incon) / n
    fpr = ff / (ff + fp) if ff + fp else 0; frc = ff / (ff + pf) if ff + pf else 0
    ff1 = 2 * fpr * frc / (fpr + frc) if fpr + frc else 0
    ppr = pp / (pp + pf) if pp + pf else 0; prc = pp / (pp + fp) if pp + fp else 0
    pf1 = 2 * ppr * prc / (ppr + prc) if ppr + prc else 0
    return dict(n=n, cov=round(cov, 3), macro_f1=round((ff1 + pf1) / 2, 3),
                fail_p=round(fpr, 3), fail_r=round(frc, 3))


vt_rows = [vt[(d["row"], d["pid"])] for d in direct if (d["row"], d["pid"]) in vt]
print("\n=== SAME-CHECKPOINT PROBE (OpenRouter Qwen3.6-35B, first 5 Titanic repos) ===")
print("Direct-property:", metrics(direct))
print("VibeTest (AT)  :", metrics(vt_rows))
json.dump({"direct": direct, "metrics_direct": metrics(direct),
           "metrics_vibetest": metrics(vt_rows)},
          open(VT.parent / "vibetest_qwen_kaggle_emnlp2026_package" /
               "supporting_docs" / "direct_property_openrouter_qwen36_35b_repo5.json", "w"), indent=2)
print("\nsaved -> supporting_docs/direct_property_openrouter_qwen36_35b_repo5.json")
