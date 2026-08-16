# Kernels — the small-M gap, a three-version split-K journey, and where prefill actually tops out

All measurements: one Apple M3 Ultra (512 GB), 4-bit g64 build unless stated.
Ledger nodes are cited as `[I#]`/`[RA#]`/`[CA#]`/`[PA#]` — see
[LEDGER.md](LEDGER.md).

## 1. The verify curve was broken, and the exotic component was innocent

The target's forward is not flat in token count `[I7]`: S=1 29.7 ms, S=7 62.5,
S=8 77.1, S=16-32 around 105, S=64 193. Against the roofline the *small* batches
are the inefficient ones — S=8 sits **3.65x off**, S=32 1.44x, S=128 1.19x
`[I12]` — which is backwards for anything physical.

This is a hybrid model (48 GatedDeltaNet linear-attention layers + 16
full-attention layers), and the linear-attention kernels were the obvious
suspect. Measured standalone they are **flat in T** (0.28-0.37 ms for T=1..64):
the recurrent state is register-resident, so the traffic is already amortized
`[I13]`. Acquit the exotic component before blaming it.

The culprit is ordinary: **MLX `quantized_matmul` does not amortize the weight
read at small M** `[I14]`. Queue-batched: bf16 is flat (M=2..16 all at 2.00x of
M=1), while 4-bit is nearly linear — 5.28x at M=7, 6.32x at M=16; M=1 sits on
the roofline and M=7 is 2.75x off it. Identical on MLX 0.31.2 and 0.32.0. Filed
upstream as [ml-explore/mlx#4265](https://github.com/ml-explore/mlx/issues/4265).
Speculative decoding lives exactly in this region — verify widths of 4-8 tokens
are the entire point of drafting — so the "weak speculation" this stack showed
was a GEMM property, not an algorithm property `[RA5]`.

## 2. Three kernel versions, and the lesson between v2 and v3

**v1 — scalar + threadgroup staging.** Amortizes correctly, runs at 0.30-0.38x
of MLX. Dead end `[I15]`.

**v2 — `simdgroup_matrix` 8x8 MMA.** Flat in M (+12% from M=1 to M=8), and
moving the barrier from every k-tile (8) to every quantization group (64) took
it 0.29 → 0.15 ms `[I15]`. Every per-shape comparison said it should win in the
model — layer-count-weighted 1.35x, `lm_head` 3.45x, Python wrapper cost zero —
and wired into the model it ran **0.56-0.82x, slower** `[I16]`.

The dependent-chain benchmark resolved the contradiction (N=K=5120, M=7)
`[I17]`:

| | independent calls | chained calls |
|---|---|---|
| MLX | 0.0765 ms | 0.0789 ms (+3%) |
| v2 (ours) | 0.0680 ms | 0.1530 ms (+125%) |

**v2 won throughput and lost latency.** A queue-batched benchmark lets the
framework overlap independent calls, so it measures throughput and hides exactly
the quantity a decode loop is made of. "1.52x in the microbench" and "0.74x in
the model" were both true at once `[CA6]`; only the chained benchmark could say
why. Decode-path kernels must be judged on a dependent chain — each call
consuming the previous call's output — with the queue-batched figure kept as a
throughput-only datapoint `[RA7]`.

**v3 — split-K.** Eight simdgroups split K eight ways, and `x` is loaded from
device memory straight into the MMA registers, removing threadgroup staging
entirely. Chained latency 0.153 → 0.0558 ms = **1.41x vs MLX** on the metric
that matters `[I18]`. The model's verify curve flattens at constant output
(top-1 match held) `[I19]`:

| verify width | before | after |
|---|---|---|
| S=6 | 62.5 ms | 44.5 |
| S=7 | 70.5 ms | 44.6 (1.58x) |
| S=8 | 77.1 ms | 43.3 (1.78x) |

The chained crossover is **M=6** — at M=4 the kernel is 0.98-1.10x (neutral),
and dispatching it there made MTP k=3 *slower* — so the shipped gate is
M in [6,8], N >= 4096, 4-bit, group 64, with everything else falling back to MLX
`[I20]` `[PA3]`. Killswitch: `MLXLM_NO_FAST_QMM=1`. The kernel is
`code/fast_qmm.py`; it is enabled from `utils.load()` — a wiring decision with
its own story, told in [speculative.md](speculative.md).

Downstream effect: wide speculation became viable for the first time on this
stack — MTP k=7 went 0.71x → 1.21x and k=5 0.78x → 1.04x `[I21]` — and one of
the DSpark drafter's two heads (confidence gating) lost its reason to exist,
because there is no longer a wide-verify cost for it to avoid `[RA9]`.

## 3. SDPA head-dim-256 fusion — real but small

The full-attention layers run head_dim 256, above the fused-SDPA path's default
coverage. A patch extending the fused path (shipped as
`code/patches/mlx_hd256_sdpa.patch`, 26 lines, gated to K <= 3 alongside the
existing T gate; against-fp32 error no worse than the fallback it replaces)
measured: **+3.2%** on 8192-token single-chunk prefill, +0.6% at 2048 `[I53]`.
The hoped-for +25-32% on the attention lane was **rejected by measurement** —
attention is only a quarter of prefill (next section), and the unfused fallback
was less bad than assumed. Kept because it is small, correct, and free.

Two adjacent prefill findings from the same pass `[I53]`: the DSpark prefill's
last chunk can emit `num_logits=1` (TTFT −215 ms constant, +4.4% at 2048), and
depth-1 pipelining of prefill chunks gains **zero** — the compute is already
saturated — so it shipped as opt-in only, documented as a non-lever.

## 4. Prefill accounting — the ceiling was already ours

A stage-level account of a T=2048 single-chunk prefill closes with residual
<= 0.05% `[I49]`: GDN layers 74.8%, attention layers 25.2%, everything else
0.08%. Across the model, **65.5% of prefill is MLP GEMM**.

Two beliefs died against that account:

- **"GEMM has ≈15% headroom" — retired** `[I51]`. In-graph MLP GEMM measures
  22.2 TFLOPS = 99% of the bf16 engine ceiling (22.45). The "19 TF" figure that
  implied headroom came from isolated single-eval microbenchmarks, which inflate
  by **x1.26-1.28 on this stack** `[I52]`; kernel verdicts have to come from
  dependent chains or layer subsets. Whole-prefill utilization: 92.7-96.3% of
  the engine ceiling.
- **"Prefill collapses with length (436 → 324 → 270)" — mostly environment**
  `[I50]`. On a quiet box with mlx 0.32.0: 4096 → 421-429 tok/s, 8192 →
  390-412. The collapse decomposed into co-resident process contention
  (dominant), a length-dependent degradation in mlx 0.31.2 (−2 to −9%, fixed in
  0.32.0 — and the box had silently been on 0.31.2 `[I54]`), **thermal droop**
  (−8 to −9% after ≈15 min sustained load), and a real SDPA quadratic term of
  only −5.9% from 2048 to 8192.

Verdict `[PA19]`: single-box prefill is **closed** — every remaining lever sums
to 4.5-6.9%, part of it already harvested by the SDPA fusion. The breakthrough
axis is two boxes ([two-box.md](two-box.md)), which took 8192-token prefill from
427 to 733.5 tok/s on this same model.

## What transfers beyond this model

1. Price the verify curve before blaming a speculation algorithm; small-M
   quantized GEMM is a known gap (mlx#4265) and it defines the economics.
2. Decode-path kernels: dependent-chain numbers or nothing. Queue-batched wins
   can coexist with in-model losses.
3. A kernel with a shape window is only as good as the loop's residency in that
   window — see [methodology.md](methodology.md) for the histogram rule.
4. Isolated microbenchmarks inflate (here x1.26-1.28). Measure in-graph.
5. Before optimizing prefill, close the accounting; at 93-96% of the engine
   ceiling the correct next move is another box, not another kernel.
