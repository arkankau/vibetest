# Methods and Evaluation Draft

## VibeTest Evaluation Protocol

VibeTest evaluates ML pipeline repositories against natural-language properties. For each repository-property pair, the model inspects the repository, collects code or configuration evidence, and returns a structured judgment:

- `PASS`: the repository appears to satisfy the property, with supporting evidence.
- `FAIL`: the repository appears to violate the property, with supporting evidence.
- `INCONCLUSIVE`: the repository does not provide enough evidence to confidently decide either way.

Each judgment also includes a `case_score` from 0.0 to 1.0. Lower scores indicate stronger confidence that the repository satisfies the property, while higher scores indicate stronger confidence that the repository violates the property. Scores near the middle are treated as less certain and can be abstained from during selective evaluation.

## Prompt Example Conditions

We compare three static-prompt settings:

- ex0: no representative examples in the prompt.
- ex10: 10 representative failure examples in the prompt.
- ex20: 20 representative failure examples in the prompt.

The purpose of this comparison is to test whether concrete examples improve the model's ability to distinguish real property violations from weak or missing evidence.

## Selective Evaluation

Because VibeTest can return `INCONCLUSIVE`, we report performance as a function of coverage. Coverage is the fraction of repository-property cases that receive a non-inconclusive decision. Macro F1 is computed over the covered examples only.

For thresholded plots, `case_score` is used to decide which predictions are confident enough to include. The main conservative plots use the current canonical thresholding setup from the plotting scripts. Dual-threshold plots are treated as an additional analysis: predictions with sufficiently low scores are treated as `PASS`, predictions with sufficiently high scores are treated as `FAIL`, and scores in the middle are abstained as `INCONCLUSIVE`.

## Synthetic Kaggle Benchmark

The synthetic Kaggle benchmark provides labeled repository-property cases derived from controlled benchmark construction. This benchmark is useful because it gives broad coverage across tasks and properties, but the current audit shows that some labels do not perfectly match the evidence-based property standard used by VibeTest.

For this reason, synthetic Kaggle results are interpreted as conservative stress-test results. The raw synthetic ground truth is used for the main synthetic figures, and the ground-truth audit is reported separately to explain label noise.

## Real Kaggle Benchmark

The real Kaggle benchmark does not have clean synthetic labels. We therefore use a conservative audit protocol for predicted failures. A sampled audit labels predicted failures as true failures or false failures based on the model's reasoning and cited evidence. This produces the headline conservative macro F1 estimates.

A separate full-context reaudit checks some false-fail cases with repository context. These checks suggest that the conservative audit may penalize some true failures too harshly. However, because this is a smaller spot check, full-context adjusted scores are reported only as sensitivity analysis, not as headline performance.

## Baselines

Reviewer-style baselines are evaluated on synthetic Kaggle using the canonical output files. They are included to compare VibeTest against a less evidence-structured review approach.

TrainCheck was also attempted as a baseline. In the canonical synthetic files available in this workspace, TrainCheck returns all cases as `INCONCLUSIVE`, resulting in 0 coverage. Because the attempted runs were execution-fragile on the current environment, TrainCheck should be described as an attempted baseline rather than a central comparison unless it is rerun under a cleaner controlled setup.

## Reporting Principles

The paper should report:

- Synthetic results using raw synthetic ground truth, with a clear note that the audit found label/property-definition noise.
- Real Kaggle results using conservative audit metrics as the headline.
- Full-context reaudit only as sensitivity analysis.
- TrainCheck as an attempted execution-fragile baseline unless a stronger rerun is completed.

The paper should avoid claiming near-perfect real Kaggle F1 from the full-context sensitivity sample. The safer claim is that VibeTest reaches roughly 0.93 conservative macro F1 on real Kaggle, with evidence that this may be an underestimate.
