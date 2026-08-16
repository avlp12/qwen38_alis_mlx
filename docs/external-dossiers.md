# External dossiers — what the rest of the ecosystem measured, and what I took from it

I keep dossiers on external work touching this model for two reasons: an
independent stack that agrees is the cheapest reproduction available, and an
independent stack that *disagrees* is a defect detector pointed at my own
integration — twice in this campaign the disagreement was the finding. Claims
below are attributed to their sources; where I re-measured, I say so. Everything
else is their number, not mine.

## 1. `ARahim3/mlx-dspark` — the other MLX implementation of DSpark

The only other public MLX implementation of DSpark-style drafting for Apple
silicon; its v0.10.0 added Qwen3.8-27B support on 2026-08-15. This is the
richest dossier of the set — a full source diff, mutual bug reproduction, and a
port list. Ledger: `[I37]`, `[I64]`-`[I68]`, `[PA23]` in [LEDGER.md](LEDGER.md).

**Triangulation.** Their registry advertises ≈1.7x speedup for this pairing,
consistent with a third-party M3 Ultra report I had earlier failed to reconcile
(58 tok/s vs raw 34.9 = 1.66x) — circumstantial attribution (bench shape and
timing line up; no statement from the author), but the numbers cohere. On
equivalent hardware and quant (M3 Ultra, 4-bit), my post-reversal stack measures
71.9 tok/s over three English/code/math prompts against that 58 — **+24%** —
and the difference has a mechanism: their README documents that on Apple
silicon, verify cost grows with token count and caps achievable speedup at
2-3x. That is precisely the small-M `quantized_matmul` gap
([kernels.md](kernels.md), mlx#4265) my split-K kernel removes — they have no
custom Metal kernel (zero `metal_kernel` call sites), so their cap-narrowing
machinery is the optimal play *in a world that still has the slope* `[I67]`.

**Mutual reproduction of the draft-slice trap** `[I64]`. Their source documents
that this head is trained anchor-as-position-0 (DeepSpec lineage) and that the
DFlash loop bundled in the drafter repo silently collapses acceptance with it —
their measured collapse 3.42 → 1.35, mine 4.23 → 2.35, same swap. Two
independent implementations hitting the same wall from opposite directions is
how `[CA1]` closed ([speculative.md](speculative.md)).

**Their rollback design is better than mine** `[I65]`. Capture-and-rerun: record
the GDN scan inputs during the verify forward, and on partial acceptance re-run
a small per-layer recurrence over the accepted prefix (bit-exact), keeping the
accepted KV. My snapshot+pending-carry design is one they built, measured, and
*discarded* ("replay backlog": a full-model row per accepted token — degrades
exactly on low-acceptance content). Since the carry tax is what makes
low-acceptance workloads expensive, this is the one structural change that could
overturn my "speculation off for Korean" rule — it is port item 1.

**Cap economics** `[I66]`. Plugging my measured verify curve into their rate
equation gives a three-band policy rather than "always max": p >= 0.65 cap 7;
0.45-0.65 **cap 1-2** (a width-2 verify at 32.1 ms undercuts the kernel-window
44 ms — the kernel only covers widths 6-8); p <= 0.4 cap 0 (plain). Their EWMA
controller with parking and fixed-period probing is the automated version of my
manual per-language rule; they measured-and-rejected exponential backoff, which
I therefore skip.

**Port list** (`[PA23]`, priority order): (1) capture-and-rerun rollback, then
re-judge Korean; (2) a simplified cap controller (the three-band policy); (3)
in-loop EOS/stop + streaming detokenizer — structural prevention of the `[I45]`
incident class; (4) if served: lookup hybrid + joint top-p truncation of q and
p; (5) small: the fair b7/cap7-vs-b8 A/B (`[I68]` — under their s=0 convention
block 7 also yields 7 drafts, so my b8-over-b7 result was partly a draft-count
confound). **Not porting**: their wide-GEMM path (my roofline accounting says
no headroom), confidence gating (both stacks measure it off), and their
wired-limit policy (different memory regime).

## 2. AtomicChat — size-vs-quality Pareto charts (GGUF side)

[AtomicChat](https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF) publishes GGUF
quantizations of this model ranked by KL divergence against the full-precision
reference — the size-quality Pareto framing that circulated when this model
landed (their Q4_K traded blows with Unsloth's UD-Q4_K_XL: slightly smaller
footprint, slightly higher KL). It is the right framing, and the GGUF side of
the fence has practiced it longer than the MLX side.

What I adopted is the method with the power fixed up, as
`harness/kl_eval.py` — the MLX-side counterpart chart's raw-data producer. My
protocol deltas, recorded in the script header so the eventual chart states
them: **full-vocabulary exact KL** in fp32 over all 248,320 logits (no top-K
truncation approximation), non-overlapping ctx-2048 windows (theirs: 4096 — a
difference that must be stated next to any cross-chart comparison), the same
windows in the same order for both models so pairing is free, top-1 agreement
from 32K+ tokens per slice rather than a short probe, and 512-token block SE.
The resident-reference design (bf16 stays loaded, targets swap) avoids a 52 GB
logit-dump artifact at ≈4 min recompute per target. The cross-build comparison
— my three builds against the mlx-vlm community conversions — is **in
progress**; results land in `results/` when done, whichever way they point.

## 3. vLLM / Intel Arc B70 — evidence that deeper MTP pays on other stacks

The CUDA/XPU side runs this model's MTP head deeper than my shipped k=2. The
[vLLM recipe for Qwen3.8-27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) ships
`{"method": "mtp", "num_speculative_tokens": 3}` as its speculative config, and
Intel Arc Pro B70 community reports credit MTP-4 with +35-50% decode on
27B-class dense models
([B70 buyer's guide](https://llmrequirements.com/intel-arc-pro-b70-local-llm-buyer-guide));
community tuning docs for vLLM MTP on other hybrids note deeper-than-1 needs
recent fixes and gets unstable past 3
([rtx6kpro notes](https://github.com/local-inference-lab/rtx6kpro/blob/master/optimization/speculative-decoding.md)).
None of these are Apple-silicon numbers and none transfer directly — verify
economics is the whole difference ([kernels.md](kernels.md)) — but they are
consistent evidence that the *head* supports useful depth beyond 2, which is
exactly what my split-K kernel made affordable to test (k=7 went 0.71x → 1.21x
`[I21]`). This dossier is a third of the motivation for the k in {2,3,4} x
acceptance-rule sweep now running ([speculative.md](speculative.md) section 6).

## 4. llama.cpp community — the draft-length sweet spot, rediscovered per stack

The llama.cpp side of this model family converged on short draft caps: the
common recommendation for the sibling Qwen3.6 is `--spec-draft-n-max 2` — more
aggressive speculation costs more on rejection — while a long-run case study
landed on draft 5 *jointly tuned* with `p-min 0.75`, and Qwen3.8-27B users
report ≈2x decode from the built-in MTP path with no separate drafter
([discussion](https://github.com/ggml-org/llama.cpp/discussions/22473),
[case study](https://dredyson.com/my-mtp-llama-cpp-journey-with-qwen3-6-27b-a-complete-real-world-case-study-after-6-months-of-testing-what-i-learned-what-broke-and-the-proven-configuration-that-finally-delivered-60-t-s/)).
Two things carry over: the sweet spot is an *economics* result (acceptance gain
vs verify-and-rejection cost), so it must be re-derived per stack rather than
copied — mine moved from k=2-only toward wider settings the day the verify
curve flattened — and draft length interacts with the acceptance rule (their
n-max x p-min coupling is the analogue of my k x spec_temp sweep axes).

## 5. MTPLX — native-MTP runtimes as an ecosystem signal

[MTPLX](https://pypi.org/project/mtplx/) is a Mac-native runtime built entirely
around models' own MTP heads — no external drafter, exact rejection sampling at
real serving temperatures — advertising up to 2.24x at temp 0.6 / top_p 0.95 /
top_k 20, with an independent M3 Max test measuring ≈40% over plain decode and
the runtime auto-selecting speculation depth 3
([test writeup](https://www.rotecodefraktion.de/en/blog/mlx-mtp-mtplx-test-m3-max/)).
Around it, the ecosystem has started shipping MTP-head artifacts for this exact
checkpoint — mlx-community publishes the split-out MTP drafter weights
([Qwen3.8-27B-MTP-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-MTP-4bit)),
and MTPLX-optimized checkpoints of the sibling model advertise the intact heads
as the point of the build.

I read this dossier as demand-side evidence for two of this campaign's calls:
**preserving the MTP head at conversion** (a build without it is locked out of
this entire runtime class — and most public conversions drop it, including the
vision-preserving mlx-vlm community builds, which carry 0 of the 31 MTP
tensors), and **rejection-sampling acceptance at real temperatures** as the
production path rather than a greedy-only story (the fork's `spec_temp`
machinery; sweep in progress). Their 2.24x headline at depth 3 is also a
second independent hint that my k=2 default is conservative — same hypothesis
as dossier 3, testing now.
