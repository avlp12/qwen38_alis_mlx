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
| decode, 4-bit, plain | 37.6 tok/s | — | `results/out2/bench3_q4awq3m.json` |
| decode, MTP k=2 (self-spec, no extra download) | 37.6 | **50.4** (4-prompt avg) / 55.7 (en/code/math) | same |
| decode, DSpark drafter (block 8, promoted defaults) | 37.6 | **62.2** (4-prompt avg) / **71.9** (en/code/math) | same |
| prefill 8K, one box → two boxes (bitwise-identical output) | 427 tok/s | **733.5** (1.72x) | `results/bench_2box/results_2box.json` |
| prefill 32K, one box → two boxes | 388 | **733.6** (1.89x) | same + `bonus32k.json` |
| served TTFT, 8.3K-token streaming request | 20.3 s | **11.9 s** (1.705x) | `results/bench_2box/serving_verdict.json` |
| quality, 8-bit vs bf16 (corpus PPL, paired, ≈103K tokens) | — | statistically indistinguishable on en / ko / code | `results/ppl_verdict.json` |
| quality, 4-bit AWQ vs uniform 4-bit | — | better on all three slices; recovers 48.7% / 26.9% / 14.7% of the gap to 8-bit (en / ko / code) | same |

Both speculative paths are lossless (token-identical to plain greedy, checked,
not assumed). The speculative rows average four fixed prompts — chat, code,
math, Korean — because single-prompt speculative numbers overstate badly (the
same build spans 33.3 to 91.5 tok/s across those prompts; see the honesty
section for the Korean column specifically).

## The builds (Hugging Face)

Three quantized builds, all preserving the **vision tower (333 tensors,
original bf16 bytes) and the vendor MTP head (31 tensors)**:

- [avlp12/Qwen3.8-27B-Alis-MLX-8bit](https://huggingface.co/avlp12/Qwen3.8-27B-Alis-MLX-8bit) — 27.9 GB, 21.8 tok/s; indistinguishable from bf16 on every corpus slice
- [avlp12/Qwen3.8-27B-Alis-MLX-6bit](https://huggingface.co/avlp12/Qwen3.8-27B-Alis-MLX-6bit) — 21.5 GB, 27.3 tok/s; the balanced default, Korean PPL indistinguishable from bf16
- [avlp12/Qwen3.8-27B-Alis-MLX-4bit](https://huggingface.co/avlp12/Qwen3.8-27B-Alis-MLX-4bit) — 15.2 GB, 37.5 tok/s plain / 62.2 speculative; AWQ recipe; the reach build for 24-32 GB Macs, with a real, documented quality cost

On the preservation claim, stated precisely: my 2026-08-15 survey of the public
MLX builds of this model found **all 12 surveyed builds carried 0 vision
tensors** (including repos named `-vision`; 7 of 12 shipped an image
preprocessor config next to weights that cannot process an image). mlx-vlm-based
`mlx-community` conversions that preserve the tower appeared around the same
time (created 2026-08-14) and were missed by that survey's search — but those
drop the **MTP head** (0 of 31 tensors). To my knowledge these three builds are
the only ones preserving **both** subsystems, which is what the pass-through
mechanism here exists to guarantee — see
[docs/conversion-integrity.md](docs/conversion-integrity.md).

## Upstream contributions

- [ml-explore/mlx-lm PR #1735](https://github.com/ml-explore/mlx-lm/pull/1735) — fix silent Qwen3.5/3.8 corruption from a double RMSNorm shift (MTP presence misused as a format discriminator); independently reproduces [issue #1197](https://github.com/ml-explore/mlx-lm/issues/1197) / [PR #1623](https://github.com/ml-explore/mlx-lm/pull/1623) on a dense 27B, with validation data left on both threads
- [ml-explore/mlx #4265](https://github.com/ml-explore/mlx/issues/4265) — `quantized_matmul` does not amortize the weight read at small M (bf16 flat at 2.00x, 4-bit 5.28x at M=7); the finding that explains this stack's speculative-decoding economics
- [ml-explore/mlx #4253](https://github.com/ml-explore/mlx/issues/4253) — `gather_mm` silent wrong results with `sorted_indices=True` on a non-contiguous lhs (closed)
- [ml-explore/mlx #4246](https://github.com/ml-explore/mlx/issues/4246) — `gather_qmm` throughput gap for MoE-typical small groups

## Repository map

```
docs/
  LEDGER.md                the campaign's working ledger (Korean, AIF-structured;
                           every number resolves to a node here — the heart of the repo)
  conversion-integrity.md  the silently dropped vision tower, the pass-through design,
                           the MTP double-shift, and the discriminator principle
  kernels.md               small-M quantized GEMM (mlx#4265), the split-K MMA journey,
                           SDPA hd256 fusion, and the prefill accounting that closed
  speculative.md           DSpark port, the head-wiring accident, the reversed verdict,
                           the EOS incident, and the operating point that survived
  two-box.md               TB5 layer-pipelined prefill: bubble law, bitwise proof,
                           427 -> 733 tok/s, server integration
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

- **Korean is a net loss for both speculative paths** — plain 37.6 vs MTP 34.3
  and DSpark 33.3 tok/s. The cause is not yet decomposed; the operational rule
  is to run Korean workloads plain. The most promising structural fix is the
  capture-and-rerun rollback ported from mlx-dspark
  ([docs/external-dossiers.md](docs/external-dossiers.md)), unstarted.
- **The MTP sweep is in progress** — k in {2,3,4} x {greedy, rejection t0.6} x
  {240, 1024 tokens} x 4 workloads under the corrected EOS/pairing protocol,
  plus a paired bf16-vs-4bit MTP-head acceptance probe. Results will be
  committed when they land, whichever way they point; until then k=2 is the
  shipped recommendation and deeper-k gains are a hypothesis with external
  corroboration only.
- **Unmeasured / unresolved**, from the ledger: the 8-bit-target reproduction of
  the DSpark result; drafter-context growth beyond 32k (verified harmless to
  2.4k); the fair block-7-with-cap-7 vs block-8 A/B (`[I68]` — my b8-over-b7
  result was partly a draft-count confound); the converter still leaving AWQ
  MTP heads in bf16 (`[PA18]`, interim tool shipped); the KL comparison against
  the mlx-vlm community builds (running).
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
