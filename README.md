# qwen38_alis_mlx

Everything I learned porting, quantizing, and accelerating **Qwen3.8-27B** on
Apple silicon — one model, end to end: conversion integrity (the vision tower
and MTP head the standard path silently drops), a Metal kernel journey that
fixed why speculative decoding underperformed, two speculative paths and a
reversed verdict, a bitwise-exact two-box prefill split over Thunderbolt 5, and
the measurement discipline all of it cost. Where
[alis-dwq](https://github.com/avlp12/alis-dwq) holds my model-agnostic
quantization lessons, this repository holds the complete record for *this*
model: the working ledger, the topical writeups, code snapshots, the harness,
and the raw measurement JSONs.

Hardware baseline throughout: Apple M3 Ultra, 512 GB (x2 for the two-box work,
linked over TB5). Model: 27.78B hybrid (48 GatedDeltaNet + 16 full-attention
layers), 248,320 vocab, 262K context, with a 333-tensor vision tower and a
31-tensor MTP head.

## Headline numbers

All measured, all reproducible from [results/](results/) and
[harness/](harness/); protocol details in [docs/methodology.md](docs/methodology.md).

| axis | from | to | record |
|---|---|---|---|
| decode, 4-bit, plain | 37.6 tok/s | — | `results/spec_restate/greedy_regress.json` |
| decode, MTP k=2 (self-spec, no extra download) | 37.6 | **46.8** (1.24x) | same |
| decode, **gated MTP k=4 + `min_draft_p` 0.6** — the recommendation | 37.6 | **52.8** (1.40x) | same |
| decode, DSpark drafter (block 8) | 37.6 | **48.3** (1.28x) | same |
| decode, DSpark + capture-and-rerun rollback (opt-in, 2026-08-16) | 37.5 | **51.7** (1.38x) | `results/exp6_rollback/` |
| **served** decode over HTTP streaming, gated MTP k=4 (server wiring `b8a8e7c`) | 37.3 | **53.1** greedy / **47.1** t1 (1.42x / 1.28x — server tax ≈0 vs in-process) | `results/exp8_server/` |
| decode, **two boxes**: TP2 (jaccl RDMA over TB5) x gated MTP | 35.8 | **74.2** (2.07x; plain TP2 alone 1.37x fails its gate — speculation is what makes the second box pay) | `results/tp2_spike/` |
| **served** on two boxes: one TP2 stack, on-demand launch | 53.1 (one box served) | **62.9** greedy / **57.7** t1 (+18.5% / +22.4%), with 1.6x prefill in the same stack | `results/serving_full2box/` |
| decode under shipped sampling (temp 1.0 · top-p 0.95 · top-k 20), gated MTP | 37.2 | **48.1** at 240 tok (1.29x) / **45.1** at 1024 (1.22x) | `results/spec_restate/samp_240.json` / `samp_1024.json` |
| prefill 8K, one box → two boxes (bitwise-identical output) | 427 tok/s | **733.5** (1.72x) | `results/bench_2box/results_2box.json` |
| prefill 32K, one box → two boxes | 388 | **733.6** (1.89x) | same + `bonus32k.json` |
| served TTFT, 8.3K-token streaming request | 20.3 s | **11.9 s** (1.705x) | `results/bench_2box/serving_verdict.json` |
| quality, 8-bit vs bf16 (corpus PPL, paired, ≈103K tokens) | — | statistically indistinguishable on en / ko / code | `results/ppl_verdict.json` |
| quality, 4-bit AWQ vs uniform 4-bit | — | better on all three slices; recovers 48.7% / 26.9% / 14.7% of the gap to 8-bit (en / ko / code) | same |
| quality, exact full-vocab KL to bf16 — 10-build tier sweep incl. mlx-community fp4 | — | AWQ 4-bit KL 0.0654 vs uniform 0.0763 vs nvfp4 0.0962 vs mxfp4 0.1437; chart + table in [docs/kl-tiers.md](docs/kl-tiers.md) | `results/kl_out/` |

> **Correction (2026-08-16, retroactive).** The speculative rows previously read
> MTP k=2 **50.4** / DSpark **62.2** (71.9 on English prompts). Those figures
> came from a harness that did not stop at end-of-sequence — the math prompt's
> post-EOS self-copy inflated acceptance to 4.53 and carried the headline — and
> are retracted; the ruling is ledger `[J7]`/`[I78]`
> ([docs/LEDGER.md](docs/LEDGER.md)). The table now carries the EOS-cut
> protocol: long-form four-prompt set (chat/code/math/Korean), medians of 3 for
> sampled rows, stop-detection outside the timed loop. Under it the ordering
> reversed — the gated in-weights MTP path leads and DSpark trails — and the
> "no speculation for Korean" rule is repealed: gated MTP gains **+27–31% on
> Korean** under real sampling `[I79]`. Details in
> [docs/speculative.md](docs/speculative.md) §6.

Speculation stays lossless in both regimes: greedy reproduces the plain stream
(equality-checked; residual divergence is confined to verification-batch
floating-point ties that a non-speculative control reproduces at the same
positions `[I80]`), and sampled decoding preserves the client-requested
distribution exactly via truncated rejection sampling (TV ≤ 0.0014 against a
synthetic oracle, 160/160 in the greedy limit `[I76]`). The speculative rows
average four fixed prompts — chat, code, math, Korean — because single-prompt
speculative numbers are upper bounds, not results.

## The builds (Hugging Face)

Three quantized builds, all preserving the **vision tower (333 tensors,
original bf16 bytes) and the vendor MTP head (31 tensors)**:

- [avlp12/Qwen3.8-27B-Alis-MLX-8bit](https://huggingface.co/avlp12/Qwen3.8-27B-Alis-MLX-8bit) — 27.9 GB, 21.8 tok/s; indistinguishable from bf16 on every corpus slice
- [avlp12/Qwen3.8-27B-Alis-MLX-6bit](https://huggingface.co/avlp12/Qwen3.8-27B-Alis-MLX-6bit) — 21.5 GB, 27.3 tok/s; the balanced default, Korean PPL indistinguishable from bf16
- [avlp12/Qwen3.8-27B-Alis-MLX-4bit](https://huggingface.co/avlp12/Qwen3.8-27B-Alis-MLX-4bit) — 15.2 GB, 37.5 tok/s plain / 52.8 with gated MTP speculation; AWQ recipe; the reach build for 24-32 GB Macs, with a real, documented quality cost

On the preservation claim, a correction (2026-08-16): my 2026-08-15 survey reported "all 12 surveyed builds carried 0 vision tensors" — that count used the wrong key pattern (`.visual.`), which misses the `vision_tower.*` naming that mlx-vlm-family conversions use. A follow-up census of 283 MLX-tagged repos found `mlx-community` builds preserving the vision tower (333 tensors, 0 MTP), and other builds carrying **both** subsystems, the earliest published 17 hours before mine. So: no "first" and no "only". What this repo guarantees is its own verified pass-through — byte-identical vision tensors, a quantized MTP head, and end-to-end image + speculative-decoding checks — with the mechanism and receipts in [docs/conversion-integrity.md](docs/conversion-integrity.md).

## Upstream contributions

- [ml-explore/mlx-lm PR #1735](https://github.com/ml-explore/mlx-lm/pull/1735) — fix silent Qwen3.5/3.8 corruption from a double RMSNorm shift (MTP presence misused as a format discriminator); independently reproduces [issue #1197](https://github.com/ml-explore/mlx-lm/issues/1197) / [PR #1623](https://github.com/ml-explore/mlx-lm/pull/1623) on a dense 27B, with validation data left on both threads
- [ml-explore/mlx #4265](https://github.com/ml-explore/mlx/issues/4265) — `quantized_matmul` does not amortize the weight read at small M (bf16 flat at 2.00x, 4-bit 5.28x at M=7); the finding that explains this stack's speculative-decoding economics
- [ml-explore/mlx #4253](https://github.com/ml-explore/mlx/issues/4253) — `gather_mm` silent wrong results with `sorted_indices=True` on a non-contiguous lhs (closed)
- [ml-explore/mlx #4246](https://github.com/ml-explore/mlx/issues/4246) — `gather_qmm` throughput gap for MoE-typical small groups


> Kernel work now has a canonical home: [avlp12/mlx](https://github.com/avlp12/mlx) (branch `alis` = v0.32.0 + the SDPA head_dim-256 fused path, with the kernel roadmap in `docs/ALIS_KERNELS.md`). The patch in `code/patches/` is the same change; the fork is where it evolves.

## Repository map

```
docs/
  HANDOFF.md               zero-prior-knowledge handoff: what to read, what the numbers
                           are, the rules and rejections, what data already exists, what's open
  LEDGER.md                the campaign's working ledger (Korean, AIF-structured;
                           every number resolves to a node here — the heart of the repo)
  conversion-integrity.md  the silently dropped vision tower, the pass-through design,
                           the MTP double-shift, and the discriminator principle
  kernels.md               small-M quantized GEMM (mlx#4265), the split-K MMA journey,
                           SDPA hd256 fusion, and the prefill accounting that closed
  speculative.md           DSpark port, the head-wiring accident, two reversed verdicts
                           (kernel, then EOS protocol), and the operating point that survived
  two-box.md               TB5 layer-pipelined prefill: bubble law, bitwise proof,
                           427 -> 733 tok/s, server integration
  kl-tiers.md              the tier chart: exact KL vs size for 10 builds (ours + community),
                           the byte-identity cross-check, and per-Mac tier guidance
  speed-journey.md         every prefill/decode attempt in one table - baselines,
                           adoptions, opt-ins, and the six rejections with diagnoses
  methodology.md           every measurement rule, each with the incident that bought it
  external-dossiers.md     mlx-dspark, AtomicChat, vLLM/B70, llama.cpp, MTPLX — what
                           the ecosystem measured and what I took from it
code/
  models/dspark.py         MLX port of the DSpark block-diffusion drafter
  dspark_generate.py       the speculative loop (promoted defaults = the measured optimum)
  fast_qmm.py              split-K MMA kernel for small-M 4-bit GEMM (M in [6,8] gate)
  prefill_2box/            the two-box prefill module (wire/runner/server/orchestrator)
  patches/fork-vs-upstream.diff   qwen3_5.py + utils.py + quant/awq.py vs upstream main
  patches/mlx_hd256_sdpa.patch    fused-SDPA coverage for head_dim 256
harness/                   the exact scripts behind every number (fork-pinned, fail-loud)
results/                   raw measurement JSONs + index (results/README.md)
```

**Code provenance:** the runnable home of everything under `code/` is my mlx-lm
fork, [avlp12/mlx-lm](https://github.com/avlp12/mlx-lm) — run it from there;
the copies here are a point-in-time snapshot (2026-08-16, fork base `f6c30eb`
over upstream `254d153`) for reading and citation, each file carrying its origin
header. The diff under `code/patches/` is exactly
`git diff origin/main -- mlx_lm/models/qwen3_5.py mlx_lm/utils.py mlx_lm/quant/awq.py`
at snapshot time.

## Reproduce

The result tables render from this checkout alone (no model downloads):

```bash
cd results && python3 ../harness/table2.py    # build comparison (size/speed/probe)
cd results && python3 ../harness/table3.py    # uniform vs AWQ, side by side
python3 harness/analyze_2box.py results/bench_2box/results_2box.json   # 2-box accounting
```

Re-measuring from scratch needs the builds and the fork:
`harness/build_eval_corpus.py` reconstructs the exact evaluation corpus from
public sources (wikitext-2 test, Korean Wikipedia, CPython stdlib — the texts
are not committed here, deliberately: they are others' content and the script
regenerates them deterministically), then `harness/ppl_eval.py` scores each
build and `harness/measure.py` produces the `m_*.json` records. Every harness
script pins the fork on `sys.path` and **fails loudly if stock mlx-lm resolves**
— that rule was paid for; see
[docs/methodology.md](docs/methodology.md).

## Honesty section

Numbers this repository does *not* claim, and work still open:

- **The Korean verdict reversed, and the wrong rule stays on the record.**
  Under the retracted protocol I published "run Korean workloads plain"; the
  EOS-cut re-measurement repealed it — gated MTP is +34% greedy and +27–31%
  under real sampling on Korean (`[I79]`). What still holds: **DSpark alone
  stays at-or-below plain on Korean** under real sampling. Its pending-carry
  rollback is the suspected structure, and the capture-and-rerun port from
  mlx-dspark ([docs/external-dossiers.md](docs/external-dossiers.md)) remains
  the unstarted fix.
- **The MTP sweep landed, retroactively cutting my own headline** — the
  promised k x sampler x length x workload sweep completed under the corrected
  protocol, restated the speculative table (see the correction above, `[J7]`),
  promoted gated k=4 over the shipped k=2 recommendation, and found MTP-head
  precision to be a non-lever (bf16 vs 4-bit head: no significant difference,
  `[I70]`). The 8-bit-target reproduction landed with it: plain 21.8, MTP k=2
  31.1 (1.42x), DSpark 33.6 (1.54x) at its kernel-less optimum, block 4 —
  larger ratios on a slower plain, still below 4-bit plain in absolute terms
  (`results/spec_restate/bench_d_q8v.json`).
- **Cross-stack comparisons are demoted.** This repository once carried "+24%
  vs the other MLX DSpark stack"; my side of that ratio came from the retracted
  protocol, and **protocol differences dominate cross-stack speculative
  comparisons** — no such multiple survives unless both stacks run one harness
  ([docs/external-dossiers.md](docs/external-dossiers.md)).
- **Unmeasured / unresolved**, from the ledger: drafter-context growth beyond
  32k (verified harmless to 2.4k); the fair block-7-with-cap-7 vs block-8 A/B
  (`[I68]` — my b8-over-b7 result was partly a draft-count confound); the
  converter still leaving AWQ MTP heads in bf16 (`[PA18]`, interim tool
  shipped); the KL comparison against the mlx-vlm community builds (running).
- **Vision quality is functionally checked, not evaluated** — the tower is
  byte-exact and describes a shapes probe correctly; no VQA suite was run.
- Several published intermediate verdicts in this campaign were **wrong and are
  kept on the record** — the ledger's `[CA#]` nodes are the map, and
  [docs/speculative.md](docs/speculative.md) walks the largest reversal in
  full.

Everything here is offered in the same spirit as the ledger that produced it:
**provisional until independently reproduced.** The harness, raw records, and
exact protocols are in this repository precisely so that disagreement can be
checked at the protocol level — if you re-run any of this on your own M-series
hardware and get different numbers, I want to know.

## Related

- [alis-dwq](https://github.com/avlp12/alis-dwq) — the model-agnostic
  quantization method and lesson base; this campaign's cross-model rules landed
  in its
  [docs/PORTING_INTEGRITY.md](https://github.com/avlp12/alis-dwq/blob/main/docs/PORTING_INTEGRITY.md)
  and its
  [examples/qwen3.8-27b](https://github.com/avlp12/alis-dwq/tree/main/examples/qwen3.8-27b)
  case study
- [avlp12/mlx-lm](https://github.com/avlp12/mlx-lm) — the fork where all of
  this runs
- [RadixArk/Qwen3.8-27B-DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark)
  — the external drafter (separate download, separate license)

## License

MIT for the contents of this repository (documents, harness, snapshots — the
code snapshots originate from MIT-licensed mlx-lm). The model weights it
describes are Apache-2.0 (inherited from Qwen); the DSpark drafter carries its
own license — read its repository before deploying it.
