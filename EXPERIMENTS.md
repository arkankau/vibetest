## Real Bug Experiments

The experiment scripts are located under `experiments/`. The main experiments
can be run as the following to run vibetest and the baselines can be run by
changing the `--method` parameter.

```sh
# two sec. vuln. datasets
uv run experiments/vulnerabilities.py --method vibetest --dataset cwe-bench
uv run experiments/vulnerabilities.py --method vibetest --dataset bibifi
# code review baseline (Codex)
uv run experiments/vulnerabilities.py --method codex --dataset cwe-bench

# three kaggle datasets for ml properties
uv run experiments/kaggle.py --method vibetest --dataset nlp
uv run experiments/kaggle.py --method vibetest --dataset diabetic
uv run experiments/kaggle.py --method vibetest --dataset titanic
# TrainCheck + code review baseline
uv run experiments/kaggle.py --method traincheck --dataset nlp
uv run experiments/kaggle.py --method codex --dataset nlp

# iclr dataset for ml properties
uv run experiments/iclr2026.py --method vibetest

# citation hallucinations
uv run experiments/hallucination.py --method vibetest --paper-list data/hallucination/sample_100.txt
```

## Synthetic (Injected) Bug Experiments

To overcome the problem with a lack of ground truth with the real bug
experiments above, we create a synthetic bug dataset seeded by the real bugs
from above. We take clean code with no known bugs and inject different property
failures into the code based on some examples of failures from the above
experiments. This gives us a dataset of code labelled with exactly the bugs
which it contains allowing us to automatically measure a tool's precision and
recall without any costly human annotation.

### Dataset Creation
1. For each of the domains (security vuln, ml bugs, citation hallucinations) we
collect all the "clean" examples as those which are not labelled with any "FAIL"
or "INCONCLUSIVE" labels by the vibetest method. This produces the "clean" sample set.
2. For each domain, we then collect ~10 examples of property failures for each
of the properties based on the verified FAIL labeled from the vibetest method.
These create the bug examples for each domain.
3. The clean samples and bug examples for each domain should be stored in the
`synth-data/` directory to be manually inspected and verified before using them for
the rest of the experiments.
4. For each clean sample of each domain we will then sample k~Unif(0,
num-properties) and then choose a random combination of k properties which we
will inject bugs for. We will use an LLM call plus the relevant bug examples of
the property failure to inject the bug into the code and then store the buggy
code for future use. Note that the code is a full repo, so the LLM call will
need to be a call to an agent given the code repo and bug examples with a prompt
to inject a bug. Also have the LLM produce an explanation of what the property
failure that was injected was and save this as the "label" for this repo.
6. The final data should be a directory of repos with injected bugs along with a
file containing all the labels consisting of where the bug was injected and a
description of the bug.

#### Implementation Details / Assumptions (Current)
- Implementation script: `scripts/create_synthetic_bug_dataset.py`
- Output root: `synth-data/` (no writes under `data/`)
- Subcommands:
  - `collect`: builds domain-specific `clean_repos.jsonl` and `bug_examples.jsonl`
  - `plan`: samples per-clean-repo property combinations into `synth-data/injection_plan.jsonl`
  - `inject`: runs agentic bug injection from `injection_plan.jsonl` (requires explicit `--confirm`)
- Current "verified FAIL" sources used by `collect`:
  - Security vulnerabilities: `*_AT-*_recall_scored.jsonl` with `verification.verified_fail=true`
  - ML bugs (Kaggle): human annotation `C` labels for `kaggle_*_AT-*` in `results/human-annotations/human_annotations_long.csv`
  - Citation hallucinations:
    - If `hallucination_AT-*` human `C` labels exist, use them as verified FAILs
    - Otherwise optional fallback (`--allow-unverified-hallucination-fallback`) uses unverified VibeTester FAIL examples to avoid empty bug-example sets
- Current "clean" operationalization used by `collect`:
  - For domains with verification labels: no verified FAIL and no INCONCLUSIVE
  - For hallucination when no AT verification labels are available: strict all-PASS and no INCONCLUSIVE (may produce zero clean repos)
- `collect` supports domain filtering via `--domains` (e.g., run only `security-vuln` and `ml-bugs`)
- `collect` supports subdataset balancing via `--equalize-clean-repos-per-subdataset`
  - This balances *clean repo counts* per subdataset (using the `dataset` field) by random downsampling to the minimum available count
  - Deterministic with `--seed`
- Bug examples are capped per property by `--max-examples-per-property` (e.g., `3`)
- `inject` is now implemented in `scripts/create_synthetic_bug_dataset.py` and requires explicit `--confirm`
  - Reads `synth-data/injection_plan.jsonl`
  - Runs a VibeTest agent per planned row to perform code edits inside sandbox
  - Requires agent to emit:
    - `/evidence/artifacts/injection_report.json`
    - `/evidence/artifacts/repo.tar.gz` (full modified repo)
  - Injection prompt explicitly requires in-place edits only (no `.bak`/backup/renamed-clean-copy artifacts)
  - Extracts modified repos into `synth-data/injected/repos/<domain>/<row>_<repo>/`
  - Computes and stores source-vs-injected diffs per sample:
    - `synth-data/injected/repos/<domain>/<row>_<repo>/injection.diff.patch`
    - `synth-data/injected/repos/<domain>/<row>_<repo>/injection.diff.json`
  - Writes labels and metadata to `synth-data/injected/labels.jsonl`
  - Writes run summary to `synth-data/injected/injection_run_summary.json`
  - Supports `--limit`, `--offset`, `--model`, `--sandbox`, `--static`, and `--stop-on-error`

### Experiment
Using the data created as described above, we then run the vibetester as well as
the baselines on the data with an automated LLM judge which determines if the
output (verdict, evidence) is correct by comparing it to the ground truth.
