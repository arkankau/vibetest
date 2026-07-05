# Synthetic Ground-Truth Audit Prompt

You are a careful human-style annotator auditing one high-confidence VibeTest prediction on the synthetic Kaggle dataset.

The synthetic dataset intentionally injects bugs into real Kaggle repositories. You will receive the tested property, the synthetic ground-truth label, the synthetic injection description when available, VibeTest's prediction/reason/evidence, and source context from the injected repository. VibeTest may agree or disagree with the synthetic label. Your job is to decide what actually happened.

Do not simply trust either the synthetic label or VibeTest's wording. Read the injected repository context and decide whether the synthetic ground truth is correct for the exact property.

## Inputs

Dataset:
{dataset}

Repository:
{repo_name}

Prompt examples condition:
{examples}

Synthetic source repository path:
{source_repo_path}

Injected repository path:
{injected_repo_path}

Property id:
{property_id}

Test property:
{test_property}

Synthetic ground-truth label:
{ground_truth_label}

Synthetic ground-truth violation description:
{ground_truth_violation_description}

VibeTest prediction:
{prediction}

VibeTest disagreement type against synthetic label:
{error_type}

VibeTest correctness against synthetic label:
{qwen_correctness}

VibeTest case score:
{case_score}

VibeTest reason:
{reason}

VibeTest evidence:
{evidence}

Injected repository source context:
{source_context}

The source context is from the injected repository, not the original clean repository. Treat it as the main evidence for this audit. VibeTest's reason/evidence may be correct, wrong, incomplete, or copied from the prompt; use it only to assess VibeTest's prediction after deciding the repository-level truth.

## Label Meaning

The synthetic ground-truth label is binary:

- 1 means the injected repository violates the property, so the correct verdict should be FAIL.
- 0 means the injected repository satisfies the property, so the correct verdict should be PASS or possibly INCONCLUSIVE, but not FAIL.

The VibeTest prediction is also binary for this audit:

- 1 means VibeTest predicted FAIL at the chosen threshold.
- 0 means VibeTest predicted PASS at the chosen threshold.

## Audit Criteria

Mark GT_CORRECT_QWEN_CORRECT when:
- The synthetic ground-truth label matches the injected repository source context.
- VibeTest's prediction also matches that correct label.
- For a ground-truth FAIL, the injected repo contains concrete evidence of a real violation of the exact property.
- For a ground-truth PASS, the injected repo does not contain a concrete violation of the exact property.

Mark GT_CORRECT_QWEN_WRONG when:
- The synthetic ground-truth label matches the injected repository source context.
- VibeTest's prediction does not match the correct label implied by the injected repository source context.
- For a ground-truth FAIL, the injected repo contains concrete evidence of a real violation of the exact property.
- For a ground-truth PASS, VibeTest's FAIL is speculative, unsupported, a best-practice complaint, about a different property, or contradicted by the source context.

Mark GT_WRONG_QWEN_CORRECT when:
- The synthetic ground-truth label does not match the injected repository source context.
- VibeTest's prediction matches the correct label implied by the injected repository source context.
- Example: synthetic label says FAIL, but the source satisfies the property and VibeTest predicts PASS.
- Example: synthetic label says PASS, but the source clearly violates the property and VibeTest predicts FAIL.

Mark GT_WRONG_QWEN_WRONG when:
- The synthetic ground-truth label does not match the injected repository source context.
- VibeTest's prediction also does not match the correct label implied by the injected repository source context.
- Example: synthetic label says FAIL for the wrong reason, but the source has no violation and VibeTest also predicts FAIL.
- Example: synthetic label says PASS, the source has a clear violation, and VibeTest predicts PASS.

Mark PROMPT_FORMAT_FAILURE when:
- VibeTest did not produce a real answer, and the main issue is output formatting or prompt echoing.
- The reason/evidence is empty, malformed, or mostly repeats the prompt instead of evaluating the repository.
- Use this label only when the source context does not make a stronger ground-truth/correctness classification obvious.

Mark AMBIGUOUS when:
- The provided source context is insufficient to decide whether the synthetic ground truth is correct.
- The property depends on runtime behavior that cannot be inferred from the shown source.
- There is conflicting evidence and no clear best label.

## Important Rules

Positive evidence matters. A FAIL ground truth requires source behavior that concretely violates the exact property.

Do not count these as failures by themselves:
- absence of an optional implementation detail
- absence of a diagnostic, sanity check, plot, or log unless the property explicitly requires it
- a general best-practice concern
- weak speculation about what might happen
- code that looks unusual but does not violate the exact property

For path/filename feature properties, a violation requires path-like or filename-derived text to actually enter model features unless the property explicitly allows it.

For class-imbalance properties, a violation requires weighting/sampling/imbalance handling to be computed from or applied to validation/test data, or otherwise leak outside training-only usage.

For shuffle properties, training shuffle is good, but validation/test shuffling is a violation when it can affect evaluation order, pairing, metric computation, or determinism. If the code has no training shuffle, that may violate the property only if the property requires training data to be shuffled.

For metric/split properties, a violation requires a concrete mismatch between the claimed/reported metric and the split or definition actually used.

For random-label, tiny-batch overfit, and loss-decrease sanity properties, distinguish between "the repo does not include this check" and "the repo includes a check that is broken." Missing a check is not automatically a violation unless the property explicitly requires adding or having that check.

For parameter-update properties, a violation requires real frozen/unupdated model parameters during training, not merely display code, temporary inference code, or optimizer setup that still includes all trainable parameters.

If VibeTest cites code that is not present in the injected source context, treat that as a VibeTest evidence problem, not automatically as synthetic ground-truth noise.

If the injected source context directly shows the injected bug described by the ground truth, the ground truth is likely correct. Then choose GT_CORRECT_QWEN_CORRECT or GT_CORRECT_QWEN_WRONG based on VibeTest's prediction.

## Output Labels

Use exactly one:

- GT_CORRECT_QWEN_WRONG: The synthetic ground-truth label is correct and VibeTest's prediction does not match it.
- GT_CORRECT_QWEN_CORRECT: The synthetic ground-truth label is correct and VibeTest's prediction is also correct.
- GT_WRONG_QWEN_CORRECT: The synthetic ground-truth label appears wrong and VibeTest's prediction matches the injected repository.
- GT_WRONG_QWEN_WRONG: The synthetic ground-truth label appears wrong and VibeTest's prediction is also wrong relative to the injected repository.
- PROMPT_FORMAT_FAILURE: The main issue is malformed/prompt-echo output, not a substantive repository judgment.
- AMBIGUOUS: The available context is not enough to decide.

## Output Format

Return only this JSON object, with no markdown:

{
  "audited_outcome": "GT_CORRECT_QWEN_CORRECT | GT_CORRECT_QWEN_WRONG | GT_WRONG_QWEN_CORRECT | GT_WRONG_QWEN_WRONG | PROMPT_FORMAT_FAILURE | AMBIGUOUS",
  "confidence": 0.0,
  "ground_truth_assessment": "One to three concise sentences explaining whether the synthetic label matches the injected repo.",
  "qwen_prediction_assessment": "Briefly explain whether VibeTest's prediction matches the repository-level truth, or why its output is malformed/ambiguous.",
  "source_evidence_check": "State the exact source behavior visible in the injected repo context that supports your decision.",
  "correct_label": 0,
  "correct_verdict": "PASS | FAIL | INCONCLUSIVE",
  "clean_human_annotation": "If the correct verdict is FAIL, write a clean human-style failure reason/evidence. If not FAIL, write an empty string."
}

## Calibration Examples

GT_CORRECT_QWEN_WRONG:
- Ground truth says FAIL for "no path/filename text as a feature." The injected source creates `SourcePath` and includes dummy columns derived from it in the model features. VibeTest says PASS. The synthetic label is correct and VibeTest's prediction is incorrect.
- Ground truth says PASS for "training data shuffles, val/test do not." The source has ordinary full-batch training without validation/test shuffling. VibeTest says FAIL only because there is no training DataLoader shuffle. If the property does not require a DataLoader specifically, this is a Qwen false alarm, not ground-truth noise.

GT_CORRECT_QWEN_CORRECT:
- Ground truth says FAIL for "no test-to-train leakage." The injected source concatenates train and test before fitting preprocessing, and VibeTest predicts FAIL. Both the ground truth and Qwen are correct.
- Ground truth says PASS for "all model parameters are updated." The source shows `optimizer = Adam(model.parameters())` and no frozen parameters, and VibeTest predicts PASS. Both the ground truth and Qwen are correct.

GT_WRONG_QWEN_CORRECT:
- Ground truth says FAIL because the injection description claims validation shuffling, but the injected source context shows only training data is shuffled and validation/test order is preserved. VibeTest predicts PASS. The ground truth is wrong and Qwen is correct.
- Ground truth says PASS, but the injected source clearly concatenates train and test before fitting preprocessing used for training. VibeTest predicts FAIL. The ground truth is wrong and Qwen is correct.

GT_WRONG_QWEN_WRONG:
- Ground truth says FAIL, but the source context does not show a real violation. VibeTest also predicts FAIL using speculative or unrelated evidence. The ground truth and Qwen are both wrong relative to the injected repo.
- Ground truth says PASS, but the source context clearly violates the property. VibeTest predicts PASS. The ground truth and Qwen are both wrong relative to the injected repo.

PROMPT_FORMAT_FAILURE:
- VibeTest reason is empty or only says "Here is the test case and repository..." with no repository analysis. If the source context does not let you confidently audit the synthetic label, classify the case as prompt/format failure.

AMBIGUOUS:
- The property depends on runtime metric determinism, but the provided source context omits the evaluation function and metric computation.
