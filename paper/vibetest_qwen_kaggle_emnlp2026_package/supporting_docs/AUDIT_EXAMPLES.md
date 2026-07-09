# Audit Examples

This note collects representative audit examples for the paper narrative. These are not meant to replace the full audit CSVs; they are short, readable examples that explain what the aggregate numbers mean.

## Synthetic Ground-Truth Audit Examples

The synthetic audit found 53 usable high-confidence disagreements:

| Outcome | Count |
|---|---:|
| Real Qwen miss | 29 |
| Ground-truth/property-definition mismatch | 24 |

The main point is that synthetic Kaggle remains useful, but raw F1 is pessimistic/noisy because some disagreements are benchmark-label issues rather than model mistakes.

### Example 1: Real Qwen miss, frozen parameters

- Dataset: diabetic
- Repository: `SYN-D1`
- Property: all model parameters should be updated during training unless freezing is explicitly intended.
- Synthetic label: `FAIL`
- VibeTest prediction: `PASS`
- Audit outcome: ground truth correct, Qwen wrong.

Why the audit kept the failure:

The notebook freezes every model parameter with `for p in learn.model.parameters(): p.requires_grad = False` before calling `learn.fit_one_cycle(1, max_lr = 2e-3)`. The model is later unfrozen, but that does not remove the earlier training stage where all parameters were frozen. VibeTest treated the later unfreeze as enough to pass, so this is a real miss.

### Example 2: Ground-truth/property mismatch, harmless loops

- Dataset: diabetic
- Repository: `SYN-D1`
- Property: avoid explicit loops or Python control flow when matrix operations could implement the same operation.
- Synthetic label: `FAIL`
- VibeTest prediction: `PASS`
- Audit outcome: ground truth wrong, Qwen correct.

Why the audit rejected the synthetic failure:

The explicit loops in the notebook are for file printing, path-string construction, and input-shape control flow. The actual numerical computation uses tensor operations such as `torch.zeros(...)` and vector assignment. The audit did not find an elementwise numerical Python loop that should be replaced by matrix operations, so the synthetic failure label was too broad.

### Example 3: Ground-truth/property mismatch, validation augmentation

- Dataset: diabetic
- Repository: `SYN-D2`
- Property: augmentation should only be applied to the training dataset; validation/test preprocessing should be deterministic.
- Synthetic label: `PASS`
- VibeTest prediction: `FAIL`
- Audit outcome: ground truth wrong, Qwen correct.

Why the audit accepted VibeTest:

The transform pipeline contains `transforms.RandomHorizontalFlip(p=0.5)`. The same `ImageFolder(..., transform=transform)` dataset object is then used for both the training and validation dataloaders. This means validation data receives the same stochastic augmentation as training data. VibeTest correctly identified a real violation even though the synthetic label marked the case as passing.

## Real Kaggle Audit Examples

The conservative real Kaggle audit sampled predicted failures and labeled them as true or false failures:

| Outcome | Count |
|---|---:|
| True fail | 122 |
| False fail | 30 |

These labels produce the headline conservative real Kaggle F1. A later full-context check of the 30 conservative false fails found that many were false only because the verifier did not receive the relevant notebook cells.

### Example 4: True fail, validation dataloader shuffled

- Dataset: Titanic
- Repository: `REAL-T1`
- Property: training data may be shuffled, but validation/test data should not be shuffled.
- Conservative audit outcome: true fail.

Why the audit kept the failure:

The repository constructs both loaders with `shuffle=True`: `data_loader_train = DataLoader(train, batch_size=32, shuffle=True)` and `data_loader_val = DataLoader(val, batch_size=32, shuffle=True)`. That directly violates the property because validation order should be deterministic.

### Example 5: True fail, incorrect ROC-AUC input

- Dataset: Titanic
- Repository: `REAL-T2`
- Property: reported metrics should use the correct split and exact metric definition.
- Conservative audit outcome: true fail.

Why the audit kept the failure:

The notebook computes `roc_auc_score(y_test, predictions)` where `predictions` comes from `classifier.predict(x_test)`. ROC-AUC should be computed from continuous scores or probabilities, not hard class labels. The cited code directly supports the failure.

### Example 6: Conservative false fail caused by missing context

- Dataset: Titanic
- Repository: `REAL-T3`
- Property: avoid explicit loops or Python control flow when matrix operations could implement the same operation.
- Conservative audit outcome: false fail.
- Full-context reaudit outcome: true fail.

What happened:

The conservative audit rejected the failure because the supplied source excerpt did not include the cited notebook cells. With full notebook context, the reaudit found the cited code: a `predict(input, weights)` function loops through layers with `for i in range(len(weights))`, applies matrix multiplication, and branches between `sigmoid` and `relu`. Another cell already shows the same two-layer forward pass written directly with vectorized operations. The full-context reaudit therefore accepted the original VibeTest failure.

### Example 7: Conservative false fail caused by missing context, device handling

- Dataset: Titanic
- Repository: `REAL-T4`
- Property: model and inputs should be consistently moved to one device.
- Conservative audit outcome: false fail.
- Full-context reaudit outcome: true fail.

What happened:

The conservative audit could not verify the cited `.cuda()` and `net.to(device)` lines because they were outside the provided excerpt. The full-context reaudit found the lines: the model is moved with `net.to(device)`, while training and validation steps use hardcoded `.cuda()`, and prediction mixes `.cuda()`, `.cpu()`, and a CPU tensor. The failure is real under full context.

## Interpretation

These examples support the current reporting strategy:

- Synthetic Kaggle should be presented as useful but pessimistic/noisy, because sampled disagreements include both real Qwen misses and benchmark-label/property-definition mismatches.
- Real Kaggle headline results should use the conservative audit, because it is the safer estimate.
- Full-context reaudit should be reported as sensitivity, because it shows the conservative audit likely over-penalizes some failures when the verifier lacks enough source context.
