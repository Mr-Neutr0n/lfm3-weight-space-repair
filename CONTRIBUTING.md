# Contributing

Start with `./verify` and `docs/method.md`. An issue with a failing replay, an
ambiguous score, a patch-restoration mismatch, or an independently written test
case is useful. Include the exact command and public artifact hash.

New experiments should have their own frozen protocol, selection budget, fresh
hidden test set and complete outcome record. Keep the current measurements
unchanged. Report controls and regressions beside gains. Distinguish a clearer
prompt, a weight edit, an activation hook and training.

CPU evidence replay needs no dependencies or model access. Optional GPU inference
uses `requirements-gpu.txt` and an explicit confirmation flag. Large checkpoints,
credentials and private final data stay outside contributions.
