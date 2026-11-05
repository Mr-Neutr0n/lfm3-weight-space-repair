# Attribution and modified weight artifacts

The target model is LiquidAI/LFM3-1.7B, copyright Liquid AI, Inc., distributed
under the LFM Open License v1.0. A complete copy is in `LICENSES/LFM-1.0.txt`.
Source checkpoint:
https://huggingface.co/LiquidAI/LFM3-1.7B/tree/0dc4268b561499c648c7ba7c9466bf3d8d01517a

The `.safetensors` files under `patches/` contain extracted original BF16 columns
and columns modified for this study on 5 November 2026. The adjacent JSON patch
specifies the affected tensors, column indices and scaling factor. In selected
patches, eight columns were zeroed. Before/after receipts identify exact bytes,
counts and hashes. These are modified, partial weight artifacts. They are not
an official Liquid AI release. Original and modified weight content remains
subject to the upstream LFM license, including its commercial-use terms.

Original experiment code, generated task cases, public graded measurements and
documentation are by Hari and use the root MIT license. GPT-6.1 Sol and GPT-6
Astra were used as the explicitly recorded editing agents; Hari is responsible
for the publication. No model-generated reasoning is treated as proof of a
neural mechanism.

This experiment was inspired by Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar
Khattab, "Sparks of Weight-Space Programming", November 2026:
https://rikiyat.github.io/blog/weight-space-programming/

No complete checkpoint, private final cases, private seed, or cluster credential
is included in this release.
