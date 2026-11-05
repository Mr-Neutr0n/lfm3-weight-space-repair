# Method and limits

## Question

Can a general-purpose LLM choose a small, useful stored-weight intervention in
an already trained language model without gradients or parameter fitting?
The target behaviour is one next-action tool call after a mission order changes.
The experiment does not test continuous flight control.

## Target and task

The target is the official post-trained LiquidAI/LFM3-1.7B checkpoint at revision
`0dc4268b561499c648c7ba7c9466bf3d8d01517a`, with 1,728,801,536 parameters in BF16.
It has 24 layers combining gated convolutions, sliding-window attention and
global attention. The editable MLP output matrices have shape 2048 by 7680.

Each case contains an earlier user request, a previous status-call turn, and a
verified current tool response. The system explicitly makes `active_order`
authoritative. Five families cover navigation, docking, cargo, power and cooling.
One member of each matched pair preserves the order; the other changes a value
or revokes permission. Revocation requires `hold_position(reason=mission_revoked)`.
There is no missing external knowledge to discover. The model must use the
provided state correctly.

All configurations use the checkpoint's native chat template and XML-like tool
syntax. Decoding is greedy, with repetition penalty 1.1, at most 4096 input
tokens and 2048 generated tokens. Thinking and direct use the checkpoint's
native thinking toggle. These settings define this study, rather than a claim
about every sampling configuration supported by Liquid AI.

## Intervention and feedback

Both attempts permit at most 16,384 changed scalar values in stored MLP output
columns and 32 adaptive target-model generation episodes. Each has a separate
7200-second allocated-GPU cap. The editor may inspect failures, architecture,
weights and captured activations with Python. Gradients, an optimizer, training,
parameter fitting and automated numerical search are excluded.

The ordinary program computes approximate contributions to a desired-versus-
observed token preference. The LLM interprets that evidence and selects columns
and scaling factors. The activation projection ignores later normalizations and
computations. It is a localization hint, not a causal proof. This intervention
changes stored weights; it does not install an activation-steering hook in the
evaluated model. Hooks are used only to capture discovery evidence.

Sol, using xhigh reasoning, initially captured eight practice generations. A
wrong parameter-order reference invalidated localization, but not those saved
model outputs. Eight further stock episodes corrected localization. Sol then
tested one zeroing candidate on eight cases and ran eight random-control cases.
The initially planned half-scaling candidate was not run. Both localization
versions and the amendment are retained.

Astra, using high reasoning and priority service, reused the corrected technical
tooling and prior development evidence. It captured eight stock practice cases,
tested half-scaling on eight, tested zeroing on eight, then ran eight control
cases. It selected four columns in layer 22 and four in layer 23, excluding Sol's
columns. Half-scaling did not change strict success. Zeroing repaired the stale
power value but introduced a noncanonical safe-hold reason. Its strict score
remained 7/8. The frozen lexicographic rule preferred it because substantive
stale-action failures decreased. This was not a net practice-score improvement.

Each random control matches the selected edit's layer allocation, column count
and factor. The perturbation norms are approximately similar, not exactly
matched. One control does not establish that selected columns are uniquely
important. The controls and tested candidates stay visible in the public data.

## Selection and independent final evaluation

Sol's first evaluation used 50 development cases and 150 originally hidden final
cases, with stock direct, edited direct, random direct and stock thinking. Its
original final results became known before the Astra retry. They are never
treated as an unseen final set for Astra.

After Astra stopped all adaptive work, the parent evaluator generated a new
150-case suite from a fresh private seed. Both patches and Astra's control were
already frozen. The same fresh cases were evaluated under stock direct, Sol
direct, Astra direct, Astra random-control direct and stock thinking. Candidate
selection, prompts and the scorer remained fixed after final feedback.

The fresh suite has new random values and 75 matched pairs from the same task
grammar. It does not contain independently written prompts, new task families,
long conversations or production tools. The editors also differ in reasoning
effort, prior evidence and candidate allocation. Comparing their patches on
the same fresh cases does not isolate editor model identity as a causal factor.

Cases and seeds stay in the local parent process. Only rendered prompts are
streamed to the GPU worker, and full prompts remain outside cluster files.
The editor was stopped before fresh cases existed and was not resumed to use
their feedback. Both processes use the same Unix identity; there is no hard OS
filesystem sandbox. This is an operational boundary, not a stronger isolation
claim.

## Scoring

Strict success requires the correct tool, exact current arguments and types,
completion inside the token cap, and the literal required hold reason. A safe
hold with another reason remains a strict failure. Supplementary categories
separate that contract error from executing a revoked action, stale numbers,
stale identifiers, identifier copying, malformed syntax and truncation.

Paired gains and regressions compare the same cases. The released exact McNemar
values are exploratory conditional tests on discordant outcomes. Shared task
templates and paired cases limit inference. A tiny gain is not evidence of a
general capability improvement, a neural mechanism, or a universal editor ranking.
Unchanged-order checks do not cover unrelated language-model capabilities.

Generated tokens measure response length, not FLOPs. Column zeroing leaves dense
matrix sizes unchanged. Runs used serial arm order, so latency is not a controlled
performance benchmark. Editor API token counts and monetary cost were unavailable
and are recorded as unknown rather than zero.

## Reproducibility boundary

`./verify` semantically rescores public practice/development outputs and recomputes
all released aggregate and paired statistics. It also checks published artifact
hashes and the changed-value counts in saved BF16 columns. Private final case
inputs, seeds, complete final responses and their token IDs are withheld. The
public final outcome rows support statistic replay, not third-party semantic
rescoring of final cases. Exact private evidence is retained with digest records.

The optional `run_gpu.py` performs fresh inference on public or caller-supplied
cases using pinned model files. It is a portable replication path, not a promise
of bit-identical behaviour across different GPUs, kernels or software versions.

The frozen worker sources in `experiments/` document the actual selection and
evaluation logic. Private cluster transport and operational receipts are excluded
from this public export. Local integrity and budget closeouts are retained.
