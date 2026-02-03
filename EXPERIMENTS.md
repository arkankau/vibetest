## Experiments

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
```
