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

> **Retired 2026-08-18, on measurement.** Merging mlx v0.32.1 and re-running the
> same A/B against stock inverts this: isolated SDPA, three alternated rotations
> with cooldowns, the patch is **12.5% slower at 2048x2048** (5.65 → 6.36 ms) and
> 6% slower at 4096, keeping only a 1.6% win at 8192. Upstream widened the
> full-attention head-dim list to 64/72/80/96/128 in that release and tuned the
> fallback this patch was competing with. Since production prefill runs 2048-token
> chunks — 8192 having been refused earlier as no better than the 2048 plateau —
> the one shape where it still wins is the one configuration this stack does not
> use, so the dispatch change is reverted upstream-of-us in `avlp12/mlx@alis`.
> The wiring was checked first: patched and stock builds return different values
> for identical input, so what was measured was the patch, not a bypass.
> **A shape-windowed optimization is a claim about the run-time distribution, and
> a compiler release can move the distribution out from under it.**
The hoped-for +25-32% on the attention lane was **rejected by measurement** —
attention is only a quarter of prefill (next section), and the unfused fallback
was less bad than assumed. Kept because it is small, correct, and free.

## 4. What the window is worth, and what sits just outside it (2026-08-18)

Measured on mlx 0.32.1 through `nn.QuantizedLinear` on a dependent chain, the
model's FFN shape (K=5120, N=17408, 4-bit g64):

| M | stock | with the kernel | gain |
|---:|---:|---:|---:|
| 6 | 0.360 ms | **0.162** | **+123%** |
| 7 | 0.473 | **0.162** | **+193%** |
| 8 | 0.454 | **0.159** | **+186%** |
| 9 | 0.604 | 0.608 | 0% — outside the window |
| 10 | 0.616 | 0.613 | +0.5% |

Two things follow. The kernel is worth far more inside its window than the
end-to-end numbers suggest — 2.2 to 2.9x on the GEMM itself, diluted by
everything else a decode step does. And **the step from M=8 to M=9 costs 3.8x**
(0.159 ms to 0.604 ms), which is the real size of the cliff that rejected gate
k=8 in §3. That rejection was never about draft economics; verify width 9 simply
falls off this edge.

The kernel accumulates into a single `simdgroup_matrix<float, 8, 8>`, which is
why `M_MAX` is 8. Widths 9-16 need a second tile accumulated alongside the
first — more registers and more MMA issue, but **the weight reads stay exactly
what M=8 already pays**. That extension is now built (§5); the weight-read
prediction held, the throughput conclusion drawn from it did not.

One honesty note on the release comparison: raw `mx.quantized_matmul` measured
flat across 0.32.0 and 0.32.1 at every M from 1 to 32 (±1-4%, noise). The
end-to-end decode gains that release brought (gated MTP +2.4%, DSpark +5.7%,
plain unchanged) are real and reproduced, but they do **not** come from the
quantized GEMM path, and the mechanism is recorded as unidentified rather than
guessed at.

Two adjacent prefill findings from the same pass `[I53]`: the DSpark prefill's
last chunk can emit `num_logits=1` (TTFT −215 ms constant, +4.4% at 2048), and
depth-1 pipelining of prefill chunks gains **zero** — the compute is already
saturated — so it shipped as opt-in only, documented as a non-lever.

## 5. The wide window: built, measured, and correctly scoped (2026-08-20)

A second accumulator covering rows 8-15, sharing the staged B tile with the
first, behind `MLXLM_FAST_QMM_WIDE=1`. The M≤8 path is untouched.

| M | stock | wide kernel | gain |
|---:|---:|---:|---:|
| 9 | 0.611 ms | **0.231** | **+164%** |
| 10 | 0.624 | 0.231 | +170% |
| 12 | 0.366 | 0.232 | +58% |
| 14 | 0.288 | 0.232 | +24% |
| 16 | 0.288 | 0.232 | +24% |

Correctness came with a lesson worth more than the kernel. Compared directly
against stock, M=9-11 differ by 6.2e-3 and it reads as a failure. Against a
shared fp32 reference the wide kernel is 3.13e-3 and stock is 3.48e-3 — ours is
marginally the more accurate of the two. The 6.2e-3 was the distance between two
roundings, not an error. An oracle's two arms have to be comparable.

**The cliff is real but it was not the binding constraint.** Deep-k was
re-adjudicated with the wide kernel on, k=4/6/8, two alternated rotations:

| k | verify width | wide off | wide on |
|---:|---:|---:|---:|
| 4 (operating point) | 5 | **53.26** | 51.99 |
| 6 | 7 | 48.42 | 48.98 |
| 8 | 9 | 47.13 | 49.11 |

k=8 loses to k=4 by 7.8% even on the most generous reading, so §3's rejection
stands. The control arms say why the generous reading is not available: k=4 and
k=6 verify at widths 5 and 7, which the wide kernel *cannot structurally
touch*, and they still moved −2.2% and +1.2%. This harness's per-cell noise is
±5-8%; k=8's "+4.1%" is not claimable. Null arms earn their seat.

A call histogram (`MLXLM_QMM_HIST=1`) explains the outcome. At k=8 the gate
truncates most drafts, so **M=9 is only 8.9% of calls** while **M=1 is 43.5%** —
the k serial drafter forwards, each a single row. Weighted by time, M=9 is 27.8%
of quantized-matmul time and the wide kernel removes 18% of it, which is real
and still invisible end to end, because the constraint is drafter latency, not
GEMM.

**And the window is not flat past 8.** M≤8 is free from 6 to 8 because it pads
into one tile; M>8 lights a genuine second tile at about 1.8x (0.127 → 0.232 ms).
The extension converts a 3.8x cliff into a 1.8x step. Relaxing DSpark's
`max_width` proves it on the model:

| arm | observed width | chat | code | math | ko | acceptance (chat/code) |
|---|---:|---:|---:|---:|---:|---|
| **8 (default)** | 8.0 | **36.41** | **49.25** | **48.66** | **33.69** | 0.765 / 1.390 |
| 12 | 12.0 | 28.78 | 37.92 | 37.27 | 25.35 | 1.051 / 1.656 |
| 16 | 16.0 | 26.16 | 40.51 | 36.15 | 25.86 | 0.805 / 1.791 |
| unbounded | 30.9 | 11.96 | 18.79 | 18.00 | 10.77 | 0.890 / 2.038 |

The default wins on all four prompts. Note the direction of the two columns:
**acceptance is highest exactly where throughput is lowest.** Acceptance is not
the objective — it only means something multiplied by the width-cost curve, and
while that curve has a step at M=8 the optimum sits in front of the step.
`max_width = 8` stays.

So the kernel ships opt-in, and the scope is the finding: **wherever the width is
ours to choose — speculative depth, block size — M≤8 wins.** The wide window is
for widths that are forced on us: batched serving with concurrent requests, and
the coverage question the maintainer asked on mlx#4265.

## 6. Prefill accounting — the ceiling was already ours

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
