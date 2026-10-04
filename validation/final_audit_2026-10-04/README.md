# Final repository audit — October 4, 2026

The saved paper models were evaluated again on CPU with the archived feature arrays. Hybrid holdout MSE, MAE and R², and five-fold CV R² means/sample standard deviations match the paper at four decimal places. The vision-only rows also match Table V at the reported precision. The maximum hybrid prediction difference from the archived MPS results is 5.159e-7 V.

This check uses the existing dataset and trained models. It does not constitute a new independent test or physical validation. The separate fresh-backbone and retraining checks on MPS/CUDA remain in the parent validation directory.

Commands, run from the repository root:

```sh
python -m physicsai.verify
python -m physicsai.train --device cpu --preflight --output runs/final_audit_2026-10-04_preflight
python -m physicsai.evaluate --device cpu --output runs/final_audit_2026-10-04_cpu
```

Use new output directory names when repeating the commands. CSV files here preserve the new metrics and differences. `dataset_checks.json` records query inclusion and label/geometry checks. `checks.json` records the comparisons, five-pass timing aggregation check and original training-protocol provenance.

The 328 scientific input/reference hashes were verified. All 27 tracked Python files parsed successfully, and the README's local file links resolved. The public repository tree and history contain no manuscript sources/PDF, author portraits, camera application or pretrained backbone weights. Earlier development history is retained separately in private author archives.

The first author approved public publication on October 4, 2026. The complete package is available as [release clagtee-2026-full-v1.0](https://github.com/Templariem/physicsai-clagtee2026-reproduction/releases/tag/clagtee-2026-full-v1.0).
