# Error Analysis

## Synthetic Disagreements

The synthetic audit should be read as an audit of property operationalization, not only as an audit of Qwen. Among 53 usable sampled high-confidence disagreements, 29 were real Qwen misses and 24 were property-definition mismatches.

The main mismatch pattern is that the injected bug label sometimes encodes a broader or narrower version of the natural-language property than the evidence rubric. For example, a synthetic `avoid replaceable loops` label can treat benign file/path loops as failures, while the VibeTest/verifier rubric treats the property as targeting active numerical loops that should be vectorized. Similarly, properties around parameter freezing or augmentation depend on whether earlier training stages, validation transforms, or stated intent count as evidence.

## Real Kaggle Audit

The conservative real audit labels sampled predicted failures without always having full repository context. When the verifier was given fuller source context, many conservative false failures became supported failures. That is why the paper should headline conservative F1 and report full-context adjusted values only as sensitivity.

## Interpretation

The clean story is not that synthetic is bad or real is easy. The clean story is that natural-language ML properties have boundaries, and benchmark labels are one operationalization of those boundaries. VibeTest exposes these boundaries because it produces evidence and abstentions rather than only binary labels.
