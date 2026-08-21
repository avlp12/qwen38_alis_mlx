# The speed journey: every attempt, in one table

Everything tried on the first quantized build's prefill and decode, in order,
with the measured outcome and the verdict — including the six rejections,
because the rejections carry the diagnoses. Every number resolves to a ledger
node in [LEDGER.md](LEDGER.md) and a raw JSON under [results/](../results/).

Baseline hardware: Apple M3 Ultra 512 GB (x2 over TB5 for the two-box rows).
Canonical decode frame: 4-bit build, four-prompt (chat/code/math/Korean)
EOS-cut long-form, dependency-chained timing, thermal alternation.

## Decode

| # | attempt | result (tok/s) | vs baseline | verdict |
|---|---|---:|---:|---|
| 0 | **first 4-bit build, plain decode** | **37.6** | 1.00x | baseline |
| 1 | split-K MMA kernel (small-M window M∈[6,8] — the speculative verify widths) | recovers the slope mlx#4265 documents | — | adopted (auto-enabled on load) |
| 2 | MTP self-speculation, k=2, ungated | 46.8 | 1.24x | superseded by #3 |
| 3 | **gated MTP: k=4 + `min_draft_p` 0.6** | **52.8** | **1.40x** | **adopted — the recommended operating point** |
| 4 | truncation-aware rejection sampling (temp 1.0 / top-p .95 / top-k 20, losslessly preserving the requested distribution) | 48.1 @240 / 45.1 @1024 | 1.29x / 1.22x | adopted |
| 5 | DSpark external block drafter (block 8, pending-carry rollback) | 48.3 | 1.28x | adopted (secondary path) |
| 6 | glue-fusion stage 1 (concat QKV/in_proj + scalar folding; bit-identical) | plain +1.5%, verify widths −5% | +1.5% | opt-in (`QWEN35_FUSED_PROJ=1`) |
| 6' | glue-fusion stages 2-3 (mega-kernels) | projected ceiling +2.4-3.1% | — | **halted at the gate** — effective launch cost re-measured at 1.2µs; `async_eval` already hides dispatch in the plain loop |
| 7 | capture-and-rerun rollback (DSpark) | 51.7 | 1.38x | opt-in; the hoped-for Korean rescue was refuted — carry's true tax is acceptance-proportional draft-slot loss |
| 8 | gate k=6/8, block-7 cap, bf16 MTP head promotion | k=8: −8.3% | — | all rejected, defaults unchanged — and the diagnosis was later corrected: see row 15, the cliff was not the binding constraint |
| 9 | **MTP head realignment to 4-bit hidden states** (on-policy self-distillation, chain loss 0.3) | operating point +6.1% (57.3 → 60.8 in its paired frame*); Korean long-form **+16.1%** | — | **adopted and published** (HF main; vendor head on the `pre-align` branch) |
| 10 | server wiring of the gated path (HTTP streaming) | 53.1 greedy / 47.1 t1 served | 1.42x, **server tax ≈ 0** | adopted (`b8a8e7c`) |
| 11 | TP2 plain (jaccl RDMA over TB5; `all_sum` 21.87µs) | 49.0 | 1.37x | **rejected** — lands exactly on the pre-registered break-even arithmetic |
| 12 | **TP2 x gated MTP composite** | **74.2** | **2.07x** | **adopted — serving integration in progress**; shrinking per-forward time amplifies speculation (+29% over the single-box MTP record) |
| 13 | TP2 stage 2 (shard the MTP block + lm_head) | 77.8 | +4.7% < +8% gate | **rejected** — decomposition proved the in-loop draft cost is chain scheduling/sync, which sharding cannot touch |
| 14 | **TP2 x gated MTP, served over HTTP** (on-demand launch of the full two-box stack) | **62.9** greedy / **57.7** t1 | +18.5% / +22.4% over the single-box served record | **adopted** — one stack also delivers ≈650 tok/s prefill; the 15% serving tax (vs ≈0% on one box) is the new frontier |
| 15 | **wide-M kernel (second MMA tile, M∈(8,16]) + deep-k re-adjudication** | M=9 GEMM 0.611 → **0.231 ms**; k=8 still 7.8% below k=4 | — | **kernel kept opt-in** (`MLXLM_FAST_QMM_WIDE=1`) — the cliff is real, the constraint is drafter latency; scope is forced widths (batched serving), not chosen ones |
| 16 | DSpark `max_width` relaxed 8 → 12 / 16 / unbounded | −21% at 12; unbounded drifts to mean width 30.9 and loses two thirds | — | **rejected** — acceptance rises 1.37x, cost rises 1.8x; `max_width = 8` stays |
| — | KV cache quantization | decode −3% | — | long-context only |

\* Row 9's absolute figures live in that harness's short-prompt 240-token
greedy frame; they are paired within it and not additive with the canonical
rows.

## Prefill (8K prompt)

| # | attempt | result (tok/s) | vs baseline | verdict |
|---|---|---:|---:|---|
| 0 | **first 4-bit build** | **≈430** | 1.00x | baseline (427.2 canonical, mlx 0.32.0; see row 6 — the baseline itself moved with the compiler) |
| 1 | fused SDPA for head_dim 256 (mlx core fork, branch `alis`) + the prefill accounting | single-box engine ceiling confirmed at 96-99% | — | adopted |
| 2 | `prefill_step_size` 8192 | no gain (2048 plateau) | — | refuted |
| 3 | **two-box layer-pipelined prefill (TB5, bitwise-identical output)** | **733** → **755** on 0.32.1 (@8K) | **1.72x**, unchanged | **adopted** (`--prefill-2box`) — see row 7: both arms moved together on the new release, so the speedup held |
| 4 | served TTFT, 8.3K-token streaming request | 20.3 s → 11.9 s | 1.705x | adopted |
| 5 | TP2 prefill (same stack as the served decode) | ≈650 (TTFT 12.8 s on 8.3K) | 1.58-1.61x | adopted for interactive serving; the layer pipeline still wins bulk prefill (755) |
| 6 | **mlx 0.32.1, with the head_dim-256 patch retired** | **441.3** one box (was 427.2) | **+3.3%** on the single-box baseline | adopted — measured on two boxes at 3x different load, both showing +2.5-2.6% against the old build in alternated A/B |
| 7 | **two-box pipeline re-measured on 0.32.1** (the number row 3 left open) | **755.2** @8K (was 733.5); 729.7 @32K (was 712.5) | **+3.0%**, ratio **1.715x** (was 1.717x) | adopted — bitwise verification re-passed on the new release first |
| 8 | ANE/CPU hybrid prefill (oMLX 0.6.x), **first verdict** | mean KL 9.8-10.2, top-1 1.6% | — | **retracted** — the harness skipped the engine's load-time warm-up of 224 ANE programs, so every program's first execution returned garbage; published as a quality collapse, and wrong |
| 9 | ANE/CPU hybrid prefill, driven through the vendor's entry sequence, their default split | **520.7** @2K, 531.8 @8K one box | **+19.6%** @8K | adopted — mean KL 0.000264, top-1 100.0% |
| 10 | **hybrid re-tuned** (`mlp .30 / gdn .375 / cpu .14 / gate .13 / down .10`) + `apply_qwen35_q4_mlp_patch` at load | **571.8** one box | **+26.0%** | **adopted** — top-1 99.95%; the loader patch alone is 9 of the 26 points, and only at chunk 2048 |
| 11 | **hybrid composed with the two-box pipeline** | **863.5** @32K (control 780.4); 748.2 @8K (control 778.2) | **+10.6%** @32K, **-3.9%** @8K | **adopted with a branch** — the levers compete rather than multiply: ANE removes compute, so link transfer dominates and the pipeline ratio falls 1.90 → 1.80 |

## The trajectory in one line

Decode: 37.6 → 52.8 single-box (53.1 served; +6.1% more at the operating
point from the realigned head) → **74.2 on two boxes (2.07x)**. Prefill: ≈430
→ 755 (1.72x) → **863.5 at 32K with the ANE hybrid composed in (2.01x)**, and
571.8 on a single box. Every rejection is preserved with its diagnosis — the two
that matter most for what comes next: dispatch is already hidden in the plain
loop (so fusion pays only inside the speculative loop), and the speculative
loop's fixed cost is scheduling, not weight reads. The wider kernel window was
built and measured (rows 15-16): it removes the M=9 cliff but does not make
deep-k economic, because the binding constraint is serial drafter latency. The
open lever that survives is draft-graph fusion.
