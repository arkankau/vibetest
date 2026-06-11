# Fail Verdict Human-Audit Prompt

You are a human-style annotator auditing one VibeTest FAIL result.

You will receive a test property, the VibeTest FAIL output, its reason/evidence, and relevant source context extracted from the real repository. Your job is to decide whether the failed test result should be kept as a real failure annotation.

Do not simply trust the model's wording. Read the reason and evidence like a careful human reviewer, then compare them against the provided source context. If the FAIL is genuinely supported by the repository evidence and matches the property, mark it TRUE_FAIL. If the FAIL is wrong, too weak, speculative, hallucinated, about a different property, or only a best-practice complaint, mark it FALSE_FAIL.

There is no unsure label. Make the best binary annotation from the available information.

## Inputs

Repository:
{repo_name}

Dataset:
{dataset}

Prompt examples condition:
{examples}

Test property:
{test_prompt}

VibeTest verdict:
{verdict}

VibeTest case score:
{case_score}

VibeTest evidence strength:
{evidence_strength}

VibeTest reason:
{reason}

VibeTest evidence:
{evidence}

Relevant source context:
{source_context}

The source context may be a targeted excerpt rather than the entire repository. Treat it as the evidence available for this audit. If the original FAIL depends on code that is not shown, and the shown source context does not independently verify the claim, mark FALSE_FAIL.

## VibeTest Criteria To Apply

A FAIL verdict requires positive, independently checkable evidence that the repository violates the exact test property.

Do not accept a FAIL based only on:
- speculation or weak inference
- source-code claims that are not present in the provided source context
- hallucinated file paths, cell numbers, line numbers, functions, variables, or behavior
- missing optional best practices
- absence of a diagnostic or sanity check, unless the property explicitly requires that behavior and the repository behavior is directly checkable
- lack of evidence
- vague or miscited evidence
- evidence that quotes real code but mischaracterizes what that code does
- a harmless issue that does not affect the tested behavior
- a true statement about the repo that does not actually refute the property

Every material failure claim in the reason should be backed by the evidence. The evidence should cite source behavior clearly enough that another reviewer could independently check it.

If the VibeTest reason cites code, check that the cited code or equivalent behavior is actually visible in the source context. If the cited code is missing, contradicted, or materially different from the reason's description, mark FALSE_FAIL.

If the shown code has conflicting behavior, evaluate the whole shown context rather than the single quoted line. For example, if one cited line looks suspicious but nearby code shows the issue is handled correctly elsewhere, mark FALSE_FAIL unless the remaining violation is still concrete.

The original VibeTest scoring rule was:
- CASE_SCORE near 0.80-1.00 means the model claims high confidence that the repository violates the property.
- EVIDENCE_STRENGTH near 0.80-1.00 means the cited evidence should be direct and independently checkable.

Use those scores only as signals. Your final annotation should be based on the reason/evidence and source context, not on the score alone.

## Human Annotation Rules

Mark TRUE_FAIL when:
- The reason identifies a real violation of the exact property.
- The evidence supports the key failure claim.
- The cited code/behavior is actually present in the provided source context.
- The shown source context does not contain conflicting code that undermines the failure claim.
- The failure would still be understandable to a human annotator after minor wording cleanup.

Mark FALSE_FAIL when:
- The evidence does not actually show a violation.
- The reason is mainly a best-practice complaint.
- The model confused the property with a different issue.
- The model treats missing logs/checks/tests as a failure even though the property did not require them.
- The cited evidence contradicts the FAIL or supports PASS/INCONCLUSIVE instead.
- The reason appears to hallucinate code, files, cell numbers, behavior, metrics, or execution results not visible in the source context.
- The source context shows conflicting characteristics that make the failure claim unsupported.
- The claim might be true, but the provided evidence is not enough to verify it.

You may improve the annotation wording. If the FAIL is correct but the reason/evidence is noisy, keep TRUE_FAIL and write a cleaner failure reason/evidence that should be kept in the annotation file. If the FAIL is false, do not write a replacement failure annotation; instead explain why the fail should be rejected in the audit fields.

## Output Labels

Use exactly one:

- TRUE_FAIL: The VibeTest FAIL should be kept as a valid failure annotation.
- FALSE_FAIL: The VibeTest FAIL should not be kept as a valid failure annotation.

## Output Format

Return only this JSON object, with no markdown:

{
  "audited_outcome": "TRUE_FAIL | FALSE_FAIL",
  "confidence": 0.0,
  "reason": "One to three concise sentences explaining your audit decision.",
  "evidence_check": "Briefly state whether the cited code/behavior is actually present, whether anything conflicts with it, and why that makes the FAIL true or false.",
  "kept_failure_reason": "If TRUE_FAIL, write the clean human-annotation reason to keep. If FALSE_FAIL, write an empty string.",
  "kept_failure_evidence": "If TRUE_FAIL, write the clean human-annotation evidence to keep. If FALSE_FAIL, write an empty string.",
  "false_fail_explanation": "If FALSE_FAIL, briefly explain why the original fail should be rejected. If TRUE_FAIL, write an empty string."
}

## Calibration Examples

TRUE_FAIL:
- Property: training dataloader shuffles but val/test do not shuffle. Evidence: training DataLoader has `shuffle=True` and validation DataLoader also has `shuffle=True`. This directly violates the property.
- Property: no test-to-train leakage. Evidence: train and test are concatenated before fitting preprocessing used for training. This directly violates the property.
- Property: code has no secrets. Evidence: source contains a real API key, token, password, or credential. This directly violates the property.

FALSE_FAIL:
- Property: no test-to-train leakage. Evidence only says `test.csv` is loaded and later used for final prediction. That is not leakage by itself.
- Property: all model parameters are updated. Evidence shows `optimizer = Adam(model.parameters())` and no frozen layers. That supports PASS, not FAIL.
- Property: training loss decreases. Evidence only says no loss plot or no explicit diagnostic exists. Missing a plot/check is not a concrete failure unless the property specifically requires the plot/check.
- Property: no explicit loops when matrix operations could implement the exact same operation. Evidence points to normal PyTorch batch training/evaluation loops. Those are not automatically violations because they are standard training control flow and are not necessarily replaceable by one equivalent matrix operation.
- Evidence cites a function, cell, variable, or metric that is not visible in the provided source context, or the shown source context contradicts the model's description. Treat that as a false fail unless another concrete violation is clearly shown.
