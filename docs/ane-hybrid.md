# The Neural Engine, measured: a +19% prefill that costs nothing, and the four days it took me to believe it

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
is two orders of magnitude below that: effectively lossless. And it is untuned —
the split above is their default, not a value we searched for.

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
