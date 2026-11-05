# LFM3 weight-space repair study

Read `README.md` for the measured outcome and `docs/method.md` before changing
task definitions, interventions, budgets, or interpretation.

Run `./verify` after code or evidence changes. It uses CPU only. GPU inference
is optional and requires the explicit `run_gpu.py --confirm-inference` flag.

Keep published results immutable. New experiments use new output directories,
fresh hidden tests, a frozen plan, and separate receipts. Preserve null results,
regressions, failed checks, and the distinction between development and final
evaluation. Report accuracy and output tokens separately.

Private final inputs, seeds, and full final prompt/output logs are excluded from
this public release. Public final outcome rows support statistic replay; public
practice/development responses also support semantic rescoring. State that limit
when describing reproducibility. Future editing must finish before hidden tests
are generated or opened.

Code and original documentation use the root MIT license. Extracted BF16 weight
artifacts remain subject to `LICENSES/LFM-1.0.txt`; retain their modification and
upstream notices. Keep credentials, caches, private cluster operations, and
unrelated experiments outside commits.

Inspect the local Git identity before commits. Hari owns the work. Do not add
an agent as a commit coauthor.
