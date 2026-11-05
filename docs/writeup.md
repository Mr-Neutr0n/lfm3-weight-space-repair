# Eight columns, two editors, one small-model repair study

We asked GPT-6.1 Sol and GPT-6 Astra to repair Liquid LFM3-1.7B by changing a
tiny part of its stored weights. Neither editor could use gradient descent,
an optimizer, parameter fitting, or training. Both had 32 adaptive model probes
and a limit of 16,384 changed BF16 values.

On the fresh hidden suite, stock direct passed 118/150, the Astra edit passed 117/150, and stock thinking passed 144/150. The Astra patch did not improve strict accuracy on this fresh suite.

This is a bounded case study. A few changed answers do not establish a general
model upgrade or a universal ranking of editor LLMs.

## The order changed. Did the model follow it?

Each synthetic mission contains an earlier request and a current, verified tool
response. The system explicitly says the current order supersedes the earlier
values. The model must make one native tool call, or hold position if permission
has been revoked. Navigation, docking, cargo, power and cooling supply five tool
families. Unchanged and updated orders form matched pairs.

This tests state use in a discrete tool decision. It does not simulate a flying
spaceship. All the information needed to answer is already in the prompt.

The stock baseline passed every unchanged development case. Errors appeared when
orders changed. Some were exact old values, some were revoked actions, and one
was an identifier-copying error. A safe hold with the wrong literal reason was
kept separate from actually acting despite revocation.

## What weight editing meant here

The editors used Python to inspect architecture, failures and captured activations.
Code computed approximate projections onto desired-versus-observed token
directions. The LLM selected columns and wrote a sparse scaling patch. That
projection is a useful hint, not a proof of a neural mechanism.

Sol, at xhigh reasoning effort, zeroed eight columns in layer 23. Astra, at high
reasoning and Fast service, zeroed four columns in layer 22 and four in layer 23.
Each edit changes 0.0009477% of the model. The evaluated intervention changes
stored weights; activation hooks were only used to collect discovery evidence.
We saved the exact original bytes, because zeroing cannot be undone by division.

Sol's first attempt repaired one extra original final mission, from 120/150 to
121/150, with no regressions. Its random control stayed at 120/150. That small
gain motivated the Astra retry, rather than a claim of a broad improvement.

Astra's half-scaling candidate left practice outcomes unchanged. Its zeroing
candidate repaired the stale power value but introduced a noncanonical safe-hold
reason. Strict practice success remained 7/8. The preset tie-break selected
zeroing because it reduced substantive stale-action failures, despite no net
strict-score gain. Its random control retained stock practice outcomes.

## A fresh hidden comparison

The first final suite had already been evaluated, so it could not serve as an
unseen Astra test. After Astra froze its patch and stopped editing, a separate
parent evaluator generated 150 fresh cases from a new private seed. Five fixed
configurations then received the same cases. Neither patch changed afterward.

| Frozen configuration | Strict success | Generated tokens | Gains / regressions vs stock |
| --- | ---: | ---: | ---: |
| Stock direct | 118/150 | 5,669 | reference |
| Sol edit | 118/150 | 5,669 | 0 / 0 |
| Astra edit | 117/150 | 5,668 | 0 / 1 |
| Astra random control | 118/150 | 5,669 | 0 / 0 |
| Stock thinking | 144/150 | 49,820 | 27 / 1 |

![Fresh hidden strict accuracy](assets/fresh-accuracy.png)

On the fresh suite, Astra changed two output sequences. One revoked action
became a safe hold with a noncanonical reason. A different, previously correct
hold acquired a noncanonical reason and became a strict failure. Actual
action-after-revocation errors decreased from five to four, while safe holds
with the wrong reason increased from three to five, among 15 revoked missions.
This is a real behaviour change with a mixed outcome. It is not a strict-score
repair. Sol changed one output sequence without a graded-outcome gain. Astra's
random control matched stock token sequences on every fresh case.

The unchanged and updated subsets, failure categories, and case-level paired
outcomes are in the released measurements. Each random control matches the
selected patch's layer allocation, column count and factor. Norms are roughly
similar, not exactly matched. One random control per attempt limits specificity
claims.

![Generated tokens per mission](assets/fresh-tokens.png)

Response tokens measure output length. A column-zeroing patch leaves the dense
matrix dimensions unchanged. We did not establish a kernel speedup or smaller
model. Serial arm order also limits latency comparisons.

## What went wrong and what stayed visible

Sol's first localization used an incorrect reference parameter order. Its
correction consumed the budget slot intended for another candidate. Astra fixed
two CPU preflight issues before consuming probes. These failures and the tested
candidates remain in the record.

During the fresh comparison, an SSH connection dropped after 655 responses had
reached the local evaluator. The existing GPU worker continued. We recovered
its saved responses, reconstructed native prompts and input IDs with the pinned
tokenizer, checked output-ID decoding and bindings, and did not rerun model
generations to recover the stream.

The editors were not in a hard OS sandbox. The parent process and directory
boundary, stopped editing process, and post-freeze creation of fresh cases are
the actual separation we used.

## How to read this result

The Astra patch did not improve strict accuracy on this fresh suite. The strict score and the semantic failure categories must both stay
visible. A safe hold with a noncanonical reason remains a strict contract
failure; we did not relax the scorer after seeing the outputs.

Both attempts localized from a small practice set and one stale numerical error.
The fresh suite varies values within the same prompt grammar. It does not test
new tool families, independently written wording, long conversations, or unrelated
language abilities. Astra reused corrected tooling and prior development
evidence, deliberately excluded Sol's columns, and used a different reasoning
effort. This is not a causal comparison of editor model identity alone.

## Recompute the public evidence

The [experiment repository](https://github.com/Mr-Neutr0n/lfm3-weight-space-repair)
contains the plans, decisions, patches, original/modified column bytes, pinned
model-file hashes, public raw practice/development responses and a CPU verification
entry point. The [Kaggle notebook](https://www.kaggle.com/code/uranium53/lfm3-weight-space-repair)
recomputes the tables from released graded outcomes without a GPU.

Private final inputs, seeds and full final responses are withheld. Final tables
can be replayed from anonymized graded outcomes; independent semantic rescoring
of those final suites is not possible from the public export. The exact private
evidence and its hashes are retained. Public practice and development responses
can be semantically rescored. Optional GPU code runs fresh inference on public
or caller-supplied cases.

Built by [Hari](https://github.com/Mr-Neutr0n), inspired by
[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/),
using [Liquid AI's LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B).
Study code, documentation and synthetic public data use MIT. Extracted weight
bytes remain under the LFM Open License v1.0. No complete checkpoint is distributed.
