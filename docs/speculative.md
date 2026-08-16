# Speculative decoding — two paths, two reversals, and the operating point that survived

Two speculative paths run on this model in the fork: **MTP self-speculation**
(the checkpoint's own 31-tensor multi-token-prediction head — no extra download)
and the **DSpark drafter** (`RadixArk/Qwen3.8-27B-DSpark`, an external 1.359B
block-diffusion drafter trained against this checkpoint). This document is the
story of how DSpark went from "rejected, 0.73x" to "the production
recommendation, 1.91x" — and then how the EOS-cut protocol audit retracted that
headline too, leaving the gated MTP path in front (§6) — with every wrong
verdict along the way kept on the record. Ledger nodes `[I#]`/`[CA#]`/`[PA#]`:
[LEDGER.md](LEDGER.md).

## 1. The port, and what parity does not prove

The drafter is block diffusion: seven slots are generated in parallel in one
forward (not autoregressively), attention is dual-source over
`[target-context K/V || block K/V]` with the block bidirectional, and the
drafter borrows the target's embedding and `lm_head` while reading the target's
intermediate hidden states at layers [4, 16, 28, 40, 52] `[I1]` `[I2]`.

The MLX port (`code/models/dspark.py`) is numerically equivalent to the PyTorch
reference: cosine 1.00000000, max abs error 1.1e-4 in float32, 1.359B parameters
matched (`harness/parity_dspark.py`) `[I3]`. The speculative loop is lossless —
token-for-token identical to plain greedy `[I5]`. And it lost end-to-end at
0.73x.

**Parity proved I had faithfully ported the wrong loop.** The `spec_generate`
bundled in the drafter repo is **DFlash's** loop (the parent method), and it
never calls DSpark's two distinguishing heads: `markov_head` (127.1M — 9.3% of
the drafter) and `confidence_head`. Both loaded; neither was ever invoked
(`grep -c "markov\|confidence" dspark_generate.py` read 0 at the time). DSpark's
real inference path lives inside SGLang and is not in the repo `[I25]`. Wiring
the Markov head took acceptance 2.23 → 3.31 on a code prompt and end-to-end
0.73x → 1.23x `[I26]`; under the card's stated conditions (block 7, temp 0.6,
English/code) acceptance then averaged 4.48 against the card's published 3.39
`[I27]` — with the caveat that my per-benchmark prompts are single "-style"
prompts, not the full suites.

A checkpoint's module list is a contract: enumerate the top-level modules and
grep the inference loop for each. A module that loads but is never called is
either dead weight or a missing feature, and the difference decides campaigns.

## 2. The draft-slice convention (`[CA1]`) — closed by an independent implementation

The reference slices drafts as `[:, -B+1:]` (each mask slot's output denoises
that slot). Implemented that way, acceptance *fell* — code 2.82 → 1.37, Korean
1.91 → 1.46 — while a one-position-shifted read (`[:, :B-1]`, the first slot
reading a *known* token's output) passed verification far better. For most of
the campaign this was an open conflict node with an unproven hypothesis attached.

Closure came from source-diffing `ARahim3/mlx-dspark` (the only other public MLX
implementation): the shifted read is not a hack, it is the **trained convention
of this head** — anchor-as-position-0, DeepSpec lineage — documented in their
`config.py`, with the same failure they measured when using the bundled DFlash
loop against it (theirs 3.42 → 1.35; mine 4.23 → 2.35 under the same swap):
**mutual reproduction** `[I64]`. The convention differs per released head (the
RedHatAI variant has no anchor), so the durable rule is to derive the draft
slice (`logits_start`) from checkpoint metadata rather than assuming either
convention.

Two more card-condition claims did not survive measurement: a bf16 drafter
bought nothing over the 4-bit one (acceptance 4.19 vs 4.23 — so the drafter
stays 0.76 GB), and the `confidence_head` measured best **off** (block 6: off
45.0 tok/s vs tau 0.10/0.25/0.50 = 37.1/38.5/42.8) `[I28]` `[I29]`. The head
exists to skip a wide block when the drafter is unsure — and once split-K
flattened verify cost across M <= 8 ([kernels.md](kernels.md)), there was
nothing left to save by narrowing, only tokens to lose `[RA9]`.

## 3. Rejection sampling: tried, insufficient at the time

Leviathan rejection sampling at temp 0.6 (ported from the fork's MTP `spec_temp`
machinery) lifted acceptance +5-8% (2.45 → 2.56 at block 7; 2.60 → 2.80 at
block 8) — short of the card's 3.39, and not enough to cross break-even on the
pre-kernel stack `[I22]`-`[I24]` `[PA6]`. The step accounting at that point
already showed verify *winning* (22 ms/token vs plain 26.7) and the loss
concentrated in the drafter's side costs `[RA8]` — which is what made the later
reversal possible once those costs were re-examined.

## 4. The reversal: three defects, all mine

The standing verdict — "MTP wins; DSpark's overhead is structural" — rested on a
≈32 ms per-step residual that had never been itemized. Re-tracing it found three
faults, none of them DSpark's `[I30]`-`[I35]`:

1. **The split-K kernel was dead code** `[I30]` `[CA7]`. `fast_qmm.enable()` had
   zero call sites outside its own file — it was live only inside the scripts
   that benchmarked it. Every production path (generate, server, both
   speculative loops) ran without it: S=8 verify at 70.1 ms instead of 43.1.
   Building a feature and not wiring it was the single largest loss of the
   campaign. Fix: `utils.load()` enables it; `MLXLM_NO_FAST_QMM=1` opts out.
2. **The verify width kept leaving the kernel's window** `[I31]`. The
   `max_pending` design let width grow to `L + n_spec`: measured histogram mean
   9.50, max 16, against a kernel window of M <= 8 — mostly in the regime
   costing 2-2.6x more. A one-line clamp (`min(n_spec, 8 - L)`) was worth
   **41.2 → 59.1 tok/s (+43%)**, width mean 7.10, max 8.
3. **Block 8 beats the trained block 7** `[I32]` `[CA8]`: over three prompts,
   b6 62.1 / b7 63.6 / **b8 71.0** / b9 70.4 / b10 68.0 / b12 66.0 tok/s. Block
   9 has *higher* acceptance than block 8 (3.55 vs 3.46) and still loses — judge
   speculation on tok/s, never on acceptance. A drafter's trained width is a
   reference value, not a ceiling `[RA12]`. (One confound found later: under the
   s=0 anchor convention, block 7 also yields 7 drafts, so my b8-vs-b7 was
   really 7-vs-6 drafts — the fair b7/cap7 A/B is still open `[I68]`.)

Two side gains: `pad_lm` (round the `lm_head` batch up to the kernel window,
4.02 → 1.45 ms, +5% — and **−2% with the kernel off**: it is welded to the
kernel, not an independent lever `[I33]`) and `defer_sync` (+1.9%, output
bit-identical `[I34]`). With all fixes, the step budget closes at 49.9 ms
predicted vs 49.0 measured — zero unexplained residual `[I35]`.

### The numbers that stood — until the protocol audit `[I36]` `[PA12]` (retracted, see §6)

Measured from stock defaults after promotion
(`max_width=8, pad_lm=True, use_conf=False, defer_sync=True, block_size=8`),
greedy, 240 tokens/prompt, four fixed prompts (chat/code/math/Korean), lossless
verified token-for-token:

| configuration | 4-prompt avg (tok/s) | en/code/math avg |
|---|---|---|
| plain | 37.63 | 37.64 |
| MTP k=2 | 50.36 (1.34x) | 55.71 (1.48x) |
| **DSpark** | **62.21 (1.65x)** | **71.86 (1.91x)** |

DSpark became the 4-bit production recommendation, MTP k=2 second — the
*opposite* of the pre-reversal verdict. The raw records are in
`results/out2/` (`bench3_*.json`, `dspark_*.json`). **This table is kept as
the record of what was published, not as current numbers: §6 restates it under
the EOS-cut protocol (`[J7]`), and the ordering reverses again.**

**What was then read as the honest exception:** on Korean alone, both paths
measured slower than plain (plain 37.6 / MTP 34.3 / DSpark 33.3), and the
operational rule became "disable speculation for Korean" `[PA13]` — a rule §6
repeals: it too was a protocol artifact.

## 5. The EOS contamination incident

An audit of the harness found it did **not stop at end-of-sequence**: every run
decoded a fixed 240 tokens, so on prompts whose answer ends early, part of the
window measured post-termination behavior. On the math prompt, 31% of the
measured window was after the answer — where one build happened to copy its own
answer (acceptance spike) and another started a fresh dialogue. That artifact
was the entirety of an apparent "-11% AWQ speculation penalty": with math
excluded, the AWQ-vs-uniform gap is MTP −0.7% / DSpark −0.6%, and on EOS-cut
paired acceptance AWQ is +5.1% `[I45]`. The other decomposed component was the
AWQ converter leaving the MTP head in bf16 (3.6-5.8 pp, fixed by
`harness/quantize_mtp.py`; converter fix tracked as `[PA18]`).

Three protocol rules were bought here (in force for every number above; see
[methodology.md](methodology.md)): cut speculative benchmarks at EOS; compare
acceptance between builds only on a common token sequence, paired; and keep
stop-detection out of the timing loop — a per-token host sync collapsed decode
37.6 → 17.2 tok/s.

Two premises also fell in the same pass: "closer to bf16 means higher
acceptance" is false — on common token streams both 4-bit builds accept *more*
than the bf16 target (q4v 3.767, AWQ 3.483, bf16 3.383) `[I46]` — and the
tap-scale-drift hypothesis died twice (global rescale moved nothing because the
drafter's `hidden_norm` absorbs it `[I42]`; per-tap rescale left 97.1% of draft
blocks unchanged `[I47]`). The planned MTP-head realignment was cancelled for
the best possible reason: there was no regression left to fix `[PA16]`.

The audit's final reach came later, and it was the largest: applied
retroactively to the §4 headline itself, the EOS-cut protocol retracted it
(`[J7]` — §6).

## 6. The sweep landed: the headline restates, the gate wins, Korean flips `[J7]`

The sweep promised here ran to completion, and its first finding was
retroactive: under the corrected protocol — EOS-cut, long-form four-prompt set
(so every scored token is genuine answer decode), medians of 3 for sampled
rows, stop-detection outside the timed loop — **the §4 headline does not
reproduce**. 62.21/50.36 were protocol artifacts, dominated by the math
prompt's post-EOS self-copy (acceptance 4.53). The ruling is ledger
`[I78]`/`[J7]`; re-running the original conditions still reproduces 60.2, so
the code did not regress — the protocol did. Restated, greedy, 4-bit build:

| configuration (greedy) | tok/s | vs plain |
|---|---|---|
| plain | 37.6 | 1.00 |
| DSpark (block 8) | 48.3 | 1.28x |
| MTP k=2 | 46.8 | 1.24x |
| **MTP k=4 + `min_draft_p` 0.6** | **52.8** | **1.40x** |

Three verdicts out of the sweep (`[I70]`-`[I79]`):

- **Ungated depth loses; the p-gate is the winning lever.** Bare k degrades
  monotonically past 2; `min_draft_p 0.6` makes k=4 the fastest configuration
  by letting uncertain stretches degenerate to near-plain steps while confident
  stretches run the full chain `[I72]`. MTP-head precision is a non-lever
  (bf16 vs 4-bit head: CIs include zero at every k `[I70]`).
- **The ordering reversed again — by protocol this time, not code.** DSpark's
  EOS-era lead evaporates EOS-cut; the gated in-weights MTP path leads it.
- **The Korean rule is repealed.** "Disable speculation for Korean" was an
  artifact of the broken protocol. Gated MTP is +34% greedy and +27–31% under
  real sampling on Korean `[I79]`. DSpark alone stays at-or-below plain there
  (pending-carry rollback structure; the capture-and-rerun port from
  mlx-dspark is the suspected fix, unstarted).

And the first **real-world numbers** — Qwen's shipped sampling defaults
(temp 1.0 · top-p 0.95 · top-k 20), with truncated rejection sampling running
target and draft through the same temperature/top-p/top-k chain, so the
requested output distribution is preserved exactly (TV ≤ 0.0014 against a
synthetic oracle; 160/160 greedy-limit match; fork commit `58ae6ec`) `[I76]`:

| configuration (temp 1.0 · top-p 0.95 · top-k 20) | 240 tok | 1024 tok |
|---|---|---|
| plain, sampler on | 37.16 (1.00) | 36.91 (1.00) |
| DSpark | 41.94 (1.13x) | 39.95 (1.08x) |
| MTP k=2 | 47.74 (1.28x) | 44.10 (1.19x) |
| **MTP k=4 + `min_draft_p` 0.6** | **48.05 (1.29x)** | **45.13 (1.22x)** |

The sampler costs plain decode only 1.2%; the greedy-to-sampled toll on the
gated path is 0.11 on the multiplier `[I77]`. The 8-bit-target reproduction
landed with the sweep: plain 21.8, MTP k=2 31.1 (1.42x), DSpark 33.6 (1.54x)
at its kernel-less optimum, block 4 — larger ratios on a slower plain, still
below 4-bit plain in absolute terms `[I73]`. Raw records:
`results/spec_restate/`.

**The operating point that survived** `[PA27]`: `mtp_num_draft_tokens=4,
min_draft_p=0.6`, adding `spec_temp=1.0, spec_top_k=20, spec_top_p=0.95` under
the shipped sampling defaults. One demotion travels with the restatement:
cross-stack speculative comparisons (the "+24%" of the mlx-dspark dossier) do
not survive it — **protocol differences dominate cross-stack comparisons**, so
treat any such delta as protocol-confounded unless both stacks run one harness
([external-dossiers.md](external-dossiers.md)).
