# The Neural Engine, measured: what a CPU/GPU/ANE prefill split actually buys

Apple ships a third compute unit on every M-series die, and oMLX 0.6.x added an
opt-in hybrid prefill that puts part of each MLP on it. The benchmarks in those
release notes are on *this* model — Qwen3.5-27B on an M3 Ultra — so the question
was not whether it applies to us but whether it survives our own measurement.

It does not, and the reasons are more interesting than the verdict.

## Decode was answered by arithmetic, not benchmarking

Decode on this build is memory-bandwidth bound, and the arithmetic closes the
question before any code runs. The 4-bit weights are 15.2 GiB, the M3 Ultra
moves 800 GB/s, so a single weight pass per token caps at 49 tok/s:

| arm | tok/s | implied bandwidth | share of the 800 GB/s roofline |
|---|---:|---:|---:|
| plain | 37.7 | 614 GB/s | 77% |
| gated MTP k=4 | 54.0 | 879 GB/s | **110%** |
| TP2 x gated MTP | 74.2 | 1208 GB/s | **151%** |

Plain decode already runs at 77% of theoretical, which is roughly the practical
ceiling, and speculation is *past* the naive roofline because it amortises one
weight read over several accepted tokens. The Neural Engine adds arithmetic, not
bandwidth, and it reaches unified memory through a narrower path than the GPU.
There is nothing here for it to win. oMLX's own notes agree — decode stays on GPU.

## Prefill: the gain is real, reproducible, and unusable

Prefill is compute-bound, so the hybrid has something to do. Tuned on a quiet
box, with the GPU-only baseline pinned at 457.2 tok/s across every candidate:

| length | GPU only | hybrid | gain |
|---:|---:|---:|---:|
| 2048 | 449.1 | 546.5 | +21.7% |
| 4096 | 457.3 | 562.6 | +23.0% |
| 8192 | 455.0 | 561.2 | +23.3% |
| 16384 | 442.7 | 537.1 | +21.3% |
| 32768 | 367.1 | 442.5 | +20.5% |

Then we measured quality, and the campaign ended:

| ANE layers | mean KL (nats) | top-1 agreement | prefill gain |
|---:|---:|---:|---:|
| 6 | 0.025 | 93.3% | +1.3% |
| 8 | 0.029 | 92.6% | +1.9% |
| 16 | 0.593 | 64.6% | +3.4% |
| 32 | 6.685 | 6.5% | +9.8% |
| **64 (the default)** | **9.747** | **1.6%** | **+22.8%** |

At the shipped setting the model's predictions agree with the unmodified model
on 1.6% of positions. For scale: the KL gap between our own quantization tiers
is 0.02-0.2 nats. This is not a degraded model, it is a different one.

## Why: INT8 is doing its job, and cosine hides the damage

The ANE runs its share in INT8. Against a common fp32 reference, on one MLP
projection:

| path | RMS relative error |
|---|---:|
| our 4-bit GPU path | 2.07e-04 |
| ANE INT8 | 8.75e-03 (**42x**) |

8.75e-03 is not a bug. For a weight whose rows have max/rms = 3.99 — ours are
Gaussian, with no outliers to speak of — symmetric per-channel INT8 predicts
`(max/rms)/(127*sqrt(12)) = 9.1e-03`. The measurement lands on the theory.

That also means the usual outlier tools have nothing to grip. Clip search over
alpha in [0.6, 1.0], scored on real captured activations, improved the output
error by **1.9%** at its best point. Per-row rescaling before handing the weight
over changed it by 0.04% — the runtime already quantizes per channel.

Upstream reports cosine similarity 0.99999 for this path, and that number is
honest: `cos ~= 1 - err^2/2` puts it at the same 4-5e-03 we measure. **The metric
is the problem.** Per-layer cosine looks like a rounding difference; sixty-four
layers of it compound into a different model. We do not evaluate approximate
paths with cosine.

## The fp16 path exists and is accurate, but the arithmetic stops paying

`qwen35_ane_compile_fp16_linear` compiles the same weight in fp16. Error drops
by an order of magnitude, to 1.9x our 4-bit path rather than 42x. The catch is
throughput. Timed against a correctly warmed baseline (1.506 ms for the full
GPU matmul):

| ANE share | INT8 | fp16 | fp16 error |
|---:|---:|---:|---:|
| 0.10 | 1.09x | 1.09x | 3.36e-04 |
| 0.15 | 1.15x | **1.15x** | 3.88e-04 |
| 0.20 | 1.22x | 1.12x | 4.34e-04 |
| 0.30 | **1.38x** | 0.75x | 5.08e-04 |

fp16 matches INT8 up to a 15% share and collapses past it — beyond that the ANE
becomes the critical path. So the honest ceiling for a quality-preserving split
is about **1.15x on the projection**, which is roughly +5% end to end. Against a
two-box layer pipeline already delivering +72%, that is not a lever worth the
dependency on a private runtime.

## The full gain was chased, not conceded

The obvious objection to stopping here is that a partial gain is not worth
taking — so the question became whether the *full* gain was reachable. Four
things were tried, and the order matters because each one narrowed what was left.

**Is the collapse a runtime bug or the precision?** Substituting per-channel
INT8 in pure MLX, with no Neural Engine involved at all, reproduces it exactly:
mean KL 9.94 and top-1 1.66%, against 9.75 and 1.54% measured on the hardware.
It is the precision.

**Can the split be chosen better?** Per-channel INT8 sensitivity varies by 296x
across a projection's output channels (4.47e-04 to 1.32e-01), and output channels
can be permuted exactly — the same permutation on `down_proj`'s input columns
leaves the layer's function unchanged, because SwiGLU is elementwise. Selecting
the channels that tolerate INT8 best cut per-layer error 2.2x at a 30% share.
**End to end it changed nothing**: KL 9.94 to 9.93 at 45%, 10.05 to 9.80 at 30%.
Past a certain exposure the model is already in the saturated regime, and halving
the per-layer error does not bring it back.

**How much would be enough?** The layer sweep says this model absorbs six to
eight layers of INT8 at this share. Reaching all sixty-four needs roughly an
order of magnitude less error per layer. Clip search offers 1.02x, channel
selection 2.2x. Only fp16 offers 10x.

**Can fp16 be made fast enough?** Its ceiling is a 15% share, because the ANE
runs fp16 at about half the GPU's rate for this shape. The runtime exposes two
physical engines, so the last lever was a dual-ANE fp16 split — which measured
**1.14x, identical to single-ANE**. `qwen35_ane_compile_fp16_linear` takes no
instance argument, so both programs land on the same engine.

So the requirement is a 10x error reduction at a 30-45% share, and what exists is
a 10x reduction at a 15% share. **Speed and quality here are the same knob pulled
in opposite directions**: the throughput comes *from* INT8, and INT8 per-channel
is exactly what a model quantized per-group-of-64 cannot absorb. The local scale
adaptation our 4-bit container already relies on is the thing the accelerator
path throws away.

Two upstream changes would reopen it — a group-wise INT8 program, or an
instance-pinned fp16 compile — and neither is something a caller can build on top
of a private runtime.

## What the upstream implementation actually does

The obvious next question is how a shipping feature can behave this way, so we
read the path rather than inferring it.

**There is no compensation mechanism.** `_compile_pair`'s `dense_slice`
dequantizes the gate and up rows to fp32 and hands them straight to the ANE
compiler — `mx.dequantize(...).astype(mx.float32)`, concatenated, nothing else.
No pre-scaling, no group-scale preservation, no correction applied on the GPU
side. For 4-bit weights the runtime path is a fused `qwen35_ane_q4_swiglu_t`
rather than the plain hybrid matmul, but the weight reaching the Neural Engine is
the same array. Our reimplementation was numerically identical to theirs.

**It is not a checkpoint mismatch either.** Their own configs describe `oQ4e` as
`{"bits": 4, "group_size": 64}` — the same granularity as ours. Whatever is lost
by flattening sixty-four per-group scales into one per-channel scale is lost on
their builds too.

**Their shipped configuration reproduces it.** Running the split from PR #2966
verbatim — ANE MLP 19%, ANE GDN 45%, CPU MLP 14%, CPU GDN 13% — with every stage
confirmed live (128 ANE operations, 2.7 s of CPU matmul):

```
mean KL 9.77    p99 21.8    top-1 agreement 1.86%
```

Routing 14% to the CPU in fp16, which lowers the INT8 exposure to 19%, does not
change the outcome.

**And the reason it went unnoticed is structural: the tuner does not measure
quality.** `omlx/admin/ane_tuning.py` computes `speedup_percent` and selects with
`min(gate_results)` — time only. There is no KL, no perplexity, no logit
comparison, no accuracy gate anywhere in the ANE tuning path. The sole quality
claim attached to the feature is the per-tensor cosine 0.99999 in the PR, and
that metric cannot see layer compounding.

That is the lesson worth carrying past this model: **an automatic tuner with no
quality gate optimizes toward destroying quality**, because the search has no
reason not to. The feature ships opt-in and off by default, which is the
mitigating fact, but a user who runs the built-in tuner and takes its
recommendation gets the fastest configuration, which is also the worst one.

One boundary on this claim: we ran their code against our checkpoint, not their
published `oQ4e` build. The mechanism is checkpoint-independent — fp32 dense to
per-channel INT8 — and their configs name group size 64, so a 4-bit group-64
build should behave the same. We have not measured theirs.

## A correctness bug worth reporting

While building our own fp16 split we found that **the first execution of an ANE
program returns garbage**. It is not the program and not the input shape:

```
first call  1.61e-01     <- wrong
second call 3.47e-04     <- correct
first program, re-run    3.47e-04     <- correct
```

A warm-up call with zeros does not clear it; a warm-up with real data does not
either. The shape of the evidence — wrong on first read, right on every later
read of the same program — points at the hybrid kernel returning before the ANE
write has landed, with the next operation acting as an accidental barrier. A
consumer that reads immediately gets stale memory. This is reproducible in a few
lines and is filed for upstream.

## Operational notes

Five separate silent no-ops cost us time on this path. Each one succeeded, logged
nothing, and simply failed to accelerate:

1. `pip install .` does not build the custom kernels — `OMLX_WITH_CUSTOM_KERNEL=1`.
2. Putting the source tree ahead of the installed package on `sys.path` imports a
   copy with no compiled `_ext`; `qwen35_ane_available()` then returns False.
3. `gdn_fraction` below `z_outputs / (z_outputs + qkv_outputs)` — 0.375 for this
   model — silently disables GDN acceleration entirely.
4. The ANE path only engages on inputs of exactly `sequence_length` flattened
   tokens, so a short prompt measures nothing.
5. Calling a patched MLP directly, outside the model's own forward, does not
   route to the ANE at all.

The rule that falls out: **when a change measures as "no difference", confirm it
ran before concluding it was harmless.** `fast.qwen35_ane_profile_snapshot()
["mlp"]["operations"]` is the counter that settles it.

Isolation is straightforward: `OMLX_BASE_PATH` redirects the whole configuration
tree, so none of this touches an existing oMLX install or its port.

Raw records under [results/exp15_ane/](../results/exp15_ane/); ledger nodes
`[I145]`-`[I159]`, `[RA44]`-`[RA55]`, `[CA29]`-`[CA31]`, `[PA51]`-`[PA52]`.
