## Experiments

The experiment scripts are located under `experiments/`. The main experiments
can be run as the following to run vibetest and the baselines can be run by
changing the `--method` parameter.

```sh
# two sec. vuln. datasets
uv run experiments/vulnerabilities.py --method vibetest --dataset cwe-bench
uv run experiments/vulnerabilities.py --method vibetest --dataset bibifi

# three kaggle datasets for ml properties
uv run experiments/kaggle.py --method vibetest --dataset nlp
uv run experiments/kaggle.py --method vibetest --dataset diabetic
uv run experiments/kaggle.py --method vibetest --dataset titanic

# iclr dataset for ml properties
uv run experiments/iclr2026.py --method vibetest
```