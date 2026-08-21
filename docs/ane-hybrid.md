# The Neural Engine, measured: a +26% prefill that costs nothing, and the four days it took me to believe it

oMLX 0.6.x ships an opt-in prefill that splits each MLP across the GPU, the
Apple Neural Engine and the CPU. Its benchmarks are on *this* model and *this*
machine, so the claim had to be checked rather than assumed.

It works. It took me a long detour to establish that, and the detour is the more
useful half of this document.

## The result

Driven the way oMLX's own engine drives it, on the author's own reference
checkpoint (`Jundot/Qwen3.8-27B-oQ4e-fp16-mtp`), at their default split
(ANE MLP 19%, ANE GDN 45%, CPU MLP 14%, CPU GDN 13%):

| prompt | GPU only | hybrid | gain |
|---:|---:|---:|---:|
| 2048 | 438.0 | 520.7 | **+18.9%** |
| 4096 | 446.2 | 535.0 | **+19.9%** |
| 8192 | 444.6 | 531.8 | **+19.6%** |
| 16384 | 433.6 | 510.3 | **+17.7%** |

Quality, measured as mean KL over all 4096 positions against the same model with
the feature off:

```
ane_mlp_ops 126 · ane_gdn_ops 96      (both stages confirmed live)
mean KL 0.000264 · p99 0.002999 · top-1 agreement 100.0%
```

For scale, the KL gap between our own quantization tiers is 0.02-0.2 nats. This
is two orders of magnitude below that: effectively lossless.

## Tuned: +26%, and a loader step that is worth 9 points of it

The table above is their default split. Re-tuning each fraction on the correct
path — one factor at a time, four rotations per point, alternating arms inside a
single process — moves the operating point to:

```
mlp_fraction 0.30 · gdn_fraction 0.375
cpu_fraction 0.14 · cpu_gate_fraction 0.13 · cpu_down_fraction 0.10
```

which gives **571.8 tok/s, +25.97%** over the same-process GPU-only arm, at
**99.95% top-1 agreement**. Two things are worth extracting from the sweep:

- `gdn_fraction` has a **cliff at its lower bound**, not a slope. The engine
  gates the GDN path on `ane_outputs < z_outputs`; below 0.375 that test fails
  and GDN silently falls back to GPU. A sweep that stops at the cliff reads the
  cliff as a plateau.
- `cpu_down_fraction` looked flat at first and I called it useless. It is not —
  it is worth +0.7%, which sits outside the 0.4% spread of a four-rotation
  measurement but inside the 4% spread of a single one. The lesson is the
  ordinary one: a factor is not flat until the noise floor is below its effect.

The larger surprise is a **loader step, not a tuning knob**. If
`apply_qwen35_q4_mlp_patch()` is not applied *before* the weights load, the same
ANE configuration yields +15.5% instead of +24.9%. Nine points of the result live
in a call that has nothing to do with the Neural Engine and everything to do with
how the MLP is laid out before it is handed over. Its effect also only appears at
chunk 2048 — at chunk 1024 it is exactly nothing (466.2 → 462.1). That detail
matters in the next section. Isolated in the two-box run, patched against
unpatched at a 32K prompt:

| | chunk 1024, 1box / 2box | chunk 2048, 1box / 2box |
|---|---:|---:|
| without the patch | 466.2 / 779.6 | 452.6 / 819.6 |
| with the patch | 462.1 / 777.3 | **480.8 / 863.5** |

At chunk 1024 it does precisely nothing. At chunk 2048 it is worth +6.2% on one
box and +5.4% on two.

## Composing with the two-box pipeline: a competition, not a product

We already had a second lever: a layer-pipelined prefill across two M3 Ultras
over Thunderbolt. The natural assumption is that the two compose — speed up each
box by 26%, keep the 1.9x pipeline ratio, collect 1.9 x 1.26. That assumption is
wrong, and it is wrong in a way worth stating generally.

Measured with both boxes patched and an ANE-off control **inside the same run**:

| configuration | 8K 1box / 2box | 32K 1box / 2box | ratio @32K |
|---|---:|---:|---:|
| ANE seq=1024, chunk 1024 | 462.1 / 743.1 | 419.7 / 777.3 | 1.85x |
| **ANE seq=2048, chunk 2048** | 507.8 / 748.2 | **480.8 / 863.5** | 1.80x |
| control, ANE off, chunk 1024 | 455.3 / 778.2 | 410.6 / 780.4 | 1.90x |

Against the same-run control, the two-box arm is **+10.6% at 32K** and **-3.9% at
8K**.

ANE improves the single box everywhere (+17.1% at 32K, +11.5% at 8K), yet the
two-box ratio *falls* (1.90 → 1.80, and 1.71 → 1.47 at 8K). The pipeline ratio is
set by the balance between compute and link transfer; ANE removes compute only,
so transfer becomes a larger share and the ratio erodes. **The two levers do not
multiply — they compete for the same headroom.**

## Where the crossover actually is, and what puts it there

The composition is positive at 32K and negative at 8K, so the serving stack has
to branch. Finding the threshold meant sweeping both chunk schedules at eight
lengths, alternated inside one process, with an offload-off control:

| N | ANE@1024 | ANE@2048 | ratio | control@1024 | control@2048 | ratio |
|---|---:|---:|---:|---:|---:|---:|
| 8192 | 765.0 | 725.8 | 0.949 | 769.0 | 693.4 | 0.902 |
| 9216 | 763.7 | 765.7 | 1.003 | | | |
| 10240 | 776.9 | 777.5 | 1.001 | 779.9 | 721.2 | 0.925 |
| **11264** | 776.8 | **799.2** | **1.029** | | | |
| 12288 | 788.4 | 808.5 | 1.026 | 791.5 | 748.5 | 0.946 |
| 16384 | 791.2 | 850.0 | 1.074 | 802.3 | 764.4 | 0.953 |
| 24576 | 791.3 | 868.6 | 1.098 | | | |
| 32768 | 773.0 | 854.9 | 1.106 | 768.4 | 753.8 | 0.981 |

The control corrects the account I gave above. I had written that the wide chunk
wins at 32K because sixteen chunks amortise its bubble where four do not. Half
right: **with the offload off the wide chunk never wins anywhere in 8K-32K.**
Decomposed against the control, two curves move in opposite directions:

| N | wide-chunk bubble cost | ANE gain @2048 | ANE gain @1024 |
|---|---:|---:|---:|
| 8192 | -9.8% | +4.7% | -0.5% |
| 10240 | -7.5% | +7.8% | -0.4% |
| 12288 | -5.4% | +8.0% | -0.4% |
| 16384 | -4.7% | +11.2% | -1.4% |
| 32768 | -1.9% | +13.4% | +0.6% |

The bubble cost does amortise with length (-9.8% → -1.9%) but never turns into a
gain on its own; what carries the wide schedule over the line is the ANE gain,
which grows the other way (+4.7% → +13.4%). The crossover is simply where the
second exceeds the first — and the model reproduces the measured ratios, tying
at 10240 where 7.8% meets 7.5%.

The third column is the other half of the design. At chunk 1024 the offload is
worth nothing at all (-0.5%, -0.4%, +0.6%) because it engages only on inputs of
exactly `sequence_length` tokens. Attached at 2048 it costs nothing when it
stands aside, so **one loaded model serves both regimes** and the branch is a
per-request choice rather than a second process.

## What we ship

The un-cached suffix picks the schedule, at a threshold of **11264 tokens**. The
crossover sits near 9216, but 9216-10240 is a tie, and switching inside a tie
earns nothing — so the threshold is the first length where the gain clears the
noise: zero opportunity cost, minimum risk of landing on the wrong side. The
wiring is
`--prefill-2box-chunk-long` with `--prefill-2box-long-tokens` (default 11264);
turning the branch *on* stays opt-in, because without the offload it is a pure
loss. One box always runs ANE seq=2048 — there is no length at which turning it
off helps a single box.

## Decode: closed by arithmetic, not benchmarking

Worth stating separately, because it needs no experiment. Decode on this build is
memory-bandwidth bound: 15.2 GiB of weights, 800 GB/s, so one weight pass per
token caps at 49 tok/s.

| arm | tok/s | implied bandwidth | share of roofline |
|---|---:|---:|---:|
| plain | 37.7 | 614 GB/s | 77% |
| gated MTP k=4 | 54.0 | 879 GB/s | **110%** |
| TP2 x gated MTP | 74.2 | 1208 GB/s | **151%** |

Plain decode is already at 77% of theoretical — about the practical ceiling — and
speculation is past the naive roofline because it amortises one weight read over
several accepted tokens. A unit that adds arithmetic through a narrower path to
the same memory has nothing to win. oMLX's notes agree: decode stays on GPU.

## The detour, and why it is worth writing down

My first measurements said the opposite: mean KL 9.8 to 10.2, top-1 agreement
1.9%, on both our build and theirs. I published that. It was wrong, and the error
was entirely mine.

**What I did.** I loaded the model with `mlx_lm.load()` and called
`enable_qwen35_ane_prefill` directly. That reaches the ANE — 128 operations
confirmed by the runtime's own profiler — so every engagement check I ran came
back positive.

**What I skipped.** oMLX's engine does two things first: it applies pre-load
patches, and it warms every compiled ANE procedure at load time. Its log says so
plainly:

```
Warmed 224 ANE procedures in 1.7s at load
Warmed the CPU sharing path on 112 modules in 3.7s at load
Eagerly compiled 64 MLP and 48 GDN procedures into 2 instance-pinned ANE programs
```

My path never printed those lines, and I never noticed they were missing.

**The part that stings.** Midway through, I isolated the underlying defect
exactly right:

```
first call   1.61e-01     <- wrong
second call  3.47e-04     <- correct
first program, re-run     3.47e-04     <- correct
```

The first execution of each ANE program returns garbage; every later execution of
the same program is fine. I wrote that up, tried a warm-up call, failed to make it
stick, and filed it as "a correctness bug we cannot work around." Upstream had
already worked around it, in the code path I had stepped around. I had the bug,
the mechanism, and the fix in front of me, and drew the opposite conclusion.

Everything downstream followed from that one substitution: a 42x error figure
against our 4-bit path, a channel-selection study, a clip search, an fp16
investigation, a dual-ANE experiment, and a verdict of "speed and quality are the
same knob." The arithmetic in each of those is still correct. The premise was not.

## The protocol that would have caught it

Reproduce oMLX behaviour through oMLX's own entry sequence, never around it:

1. `maybe_apply_pre_load_patches(model_path, model_settings=..., for_vlm=True)`
2. `mlx_vlm.utils.load(model_path)`
3. `enable_qwen35_ane_prefill(model, ...)`
4. **Confirm `Warmed N ANE procedures` appears in the log before trusting any
   measurement.**

Step 4 is the one that matters. I had already written the rule this session —
*when a change measures as "no difference", confirm it ran before concluding it
was harmless* — and I applied it to engagement (did the ANE execute?) but not to
initialisation (did it execute *correctly configured*?). A positive engagement
signal is not a positive correctness signal.

## Operational notes that survived

Five silent no-ops cost real time on this path. Each one succeeded, logged
nothing, and simply failed to do its job:

1. `pip install .` does not build the custom kernels — `OMLX_WITH_CUSTOM_KERNEL=1`.
2. Putting the source tree ahead of the installed package on `sys.path` imports a
   copy with no compiled `_ext`, and `qwen35_ane_available()` returns False.
3. `gdn_fraction` below `z_outputs / (z_outputs + qkv_outputs)` — 0.375 here —
   silently disables GDN acceleration.
4. The ANE engages only on inputs of exactly `sequence_length` flattened tokens,
   so a short prompt measures nothing.
5. `model_settings.json` needs `{"version": 1, "models": {...}}`; a flat dict
   parses fine and is ignored.

And two measurement confounds worth naming: the server's **prefix cache** makes a
repeated prompt look like 2300 tok/s of prefill (vary the prompt), and
`omlx.patches.*` log lines do not reach the server log file, so their absence
proves nothing.

Isolation is clean: `OMLX_BASE_PATH` redirects the whole configuration tree, so
none of this touches an existing install or its port.

Raw records under [results/exp15_ane/](../results/exp15_ane/); ledger nodes
`[I145]`-`[I165]`, `[RA44]`-`[RA59]`, `[CA29]`-`[CA33]`, `[PA51]`-`[PA55]` —
including the retracted ones, kept with their corrections.
