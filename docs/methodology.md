# Methodology — every rule here was paid for

Nothing in this file is hypothetical hygiene. Each rule below is stated with the
incident that bought it — what the wrong protocol read, what the corrected one
read, and how far apart they were. Ledger nodes in
[LEDGER.md](LEDGER.md); the generalized cross-model battery is in
[alis-dwq docs/PORTING_INTEGRITY.md](https://github.com/avlp12/alis-dwq/blob/main/docs/PORTING_INTEGRITY.md).

## Benchmarking

**1. Decode-path kernels are judged on a dependent chain, not a queue batch**
`[I17]` `[RA7]` `[CA6]`. Queue-batching N independent calls avoids the ≈250 us
per-call sync floor — and lets the framework overlap them, measuring throughput
while a decode loop is made of latency. The incident: a kernel that won the
microbench (and every per-shape comparison, layer-weighted 1.35x) ran the model
at 0.56-0.82x; chained, it was +125% latency where MLX was +3%. Both numbers
were true at once; only the chained bench said why. Gating thresholds come from
the chained curve too — the same kernel's crossover is M=6 chained, and a
queue-batched threshold dispatched it into a regime where it made MTP k=3
slower.

**2. Isolated microbenchmarks inflate — here by x1.26-1.28** `[I51]` `[I52]`.
A single-eval GEMM microbench claimed 15% headroom that in-graph measurement
(22.2 of 22.45 TFLOPS = 99%) proved absent. Kernel verdicts come from dependent
chains or in-graph layer subsets; the isolated figure is an upper bound on
itself, nothing more.

**3. Thermal droop is a −8 to −9% effect on this hardware** `[I50]` `[PA20]`.
Fifteen minutes of sustained load costs 8-9% on subsequent runs. Any long A/B
needs crossed ordering and cooldown control; without them, do not trust
differences under ≈9%. The two-box sweeps ran serpentine order with cooldown
sleeps and reproduced to <= 0.15% — that is what the control buys.

**4. Quiet box, or the numbers are about the box** `[I50]`. The apparent
"prefill length collapse" (436 → 324 → 270 tok/s) was mostly co-resident process
contention, with a helping of an old MLX version and thermal droop; the real
length effect was −5.9% over 2048 → 8192. Corollary paid for separately: the
harness must also *prove which library it imported* — a silent fallback to stock
mlx-lm invalidated every measurement taken before the pin-and-assert went in
(`harness/measure.py` header; the tripwire lives on in `harness/table2.py`).

**5. Start the decode timer after the first token.** Otherwise prefill folds
into decode and manufactures a "plain-decode cliff" that is not there. Encoded
in `harness/kv_measure.py` along with batching the `mx.eval` (a per-iteration
eval lays a ≈250 us sync floor under the loop).

## Speculative decoding measurement

**6. Cut at EOS** `[I45]`. The harness once decoded a fixed 240 tokens
regardless of termination; on the math prompt 31% of the window measured
post-answer behavior, where one build copied its own answer (acceptance spike)
and another started a new dialogue. That artifact was the entirety of an
apparent −11% recipe regression — corrected, the gap was −0.7%/−0.6%. What a
model does after its answer ends is luck, not performance.

**7. Acceptance comparisons between builds: common token sequence, paired**
`[I45]` `[I46]`. Each build free-running its own text measures the text as much
as the build. On a shared stream the "closer to bf16, higher acceptance" premise
inverted: both 4-bit builds accept more than the bf16 target (3.767 / 3.483 vs
3.383). `harness/accept_probe.py` and `harness/accept_eos.py` implement the
paired protocol.

**8. Keep stop-detection out of the timing loop** `[I45]`. A per-token host sync
for EOS checking collapsed plain decode 37.6 → 17.2 tok/s — the measurement
destroying the thing measured. Separate the acceptance-measuring run from the
speed-measuring run, or collect device-side and scan after the loop
(`harness/accept_eos.py`; the sweep harness marks EOS-inside-window plain runs
invalid rather than paying the sync).

**9. A single-prompt speculative number is a best case, not a result** `[PA10]`
`[I28]`. The same MTP k=2 config read 1.41x on a code prompt alone, 1.32x over
three English/code prompts, 1.10x over three including Korean. Even under the
restated EOS-cut protocol one configuration spans 32.7 to 68.3 tok/s across the
four fixed prompts (DSpark, greedy). Minimum three dissimilar workloads, report
the average, name the prompts — and remember a language slice can invert the
sign *per path*: under real sampling DSpark still loses to plain on Korean
while the gated MTP path gains 27-31% there `[I79]` — probe the slice for each
path, not once for "speculation".

**10. Judge speculation on tok/s, never on acceptance — and the trained block
width is a reference, not a ceiling** `[I32]` `[RA12]` `[CA8]`. Block 9 accepts
more than block 8 (3.55 vs 3.46) and is slower (70.4 vs 71.0 tok/s); block 8
beats the drafter's trained block 7 (71.0 vs 63.6). Where verify cost is flat,
spend the width; sweep it instead of inheriting the card's number.

## Statistics

**11. Check probe power before believing a probe — a 62-token probe read the
sign backwards** `[I38]` `[I44]` `[CA9]` `[CA10]`. The early quality probe
scored 62-112 tokens per slice, so one token moved 0.89-1.61 pp; its English
standard error (±0.0406) was 3.6x the effect it claimed, and it reported the
AWQ-vs-uniform English difference with the **wrong sign** (+0.0697 vs the true
−0.01127). The published card carried that table until it was replaced
(`[PA14]`) by corpus-scale strided PPL: ≈103K scored tokens, paired per token,
blockwise SE (512-token blocks) — 16-20x the sensitivity. Top-1-agreement
probes cannot rank recipes within a bit tier; they are short by two orders of
magnitude.

**12. Assert your lengths; mark short runs invalid.** The PPL/KL harnesses
assert the corpus yields enough windows before scoring
(`harness/kl_eval.py`); the sweep harness pre-validates that prompts survive
the full measurement window and marks any run that terminates early as invalid
instead of quietly averaging a truncated segment. A length you did not check is
a denominator you do not know.

## Systems discipline

**13. A feature you built but never wired does not exist** `[I30]` `[CA7]`.
`fast_qmm.enable()` had zero call sites outside its own file; every production
path ran without the kernel for most of the campaign, and a published verdict
rested on the un-fixed stack. The grep oracle: search the repo for the entry
point of anything you just built — zero hits outside its definition means it is
dead code, whatever its benchmark said. Wire at the loading boundary
(`utils.load()`), keep a killswitch env var. Same family: a checkpoint's module
list is a contract — a loaded-but-never-called module (`markov_head`, `[I25]`)
is a missing feature until proven dead weight.

**14. An optimization with a shape window is a claim about the run-time
distribution** `[I31]`. The kernel wins at M <= 8; the loop ran at mean width
9.50, max 16 — mostly in the 2-2.6x-worse regime. Histogram the shape the loop
actually runs at; a one-line clamp into the window was +43%. And re-measure any
welded side-lever when the main feature's status changes: `pad_lm` is +5% with
the kernel on and −2% with it off `[I33]`.

**15. An unaccounted overhead is an unmeasured item** `[I35]` `[RA11]`. The
"structural 32 ms drafter overhead" that justified a shipped verdict was two of
my own defects in a costume; itemized, the step budget closed at 49.9 ms
predicted vs 49.0 measured. Require the budget to close before publishing any
verdict that rests on a residual.

## Composing two optimizations

**Never multiply two separately measured gains.** Our ANE offload was worth
+17.1% on one box and our two-box pipeline 1.90x; the composition delivered
+10.6%, and at a shorter prompt **-3.9%**. Both parts were measured correctly.
The product was still fiction, because the levers attack different bottlenecks
(compute and link transfer) and relieving one promotes the other — the pipeline
ratio itself fell to 1.80. Measure the composition, with a control arm for the
second lever **inside the same run**.

**Expect a crossover and find the parameter that moves it.** A composition that
is positive at one operating point and negative at another is the normal case,
not a measurement failure. Ours moved with prompt length through chunk count.
Publish the branch, not the favourable half.

**When a vendor's number beats yours on an identical configuration, look
upstream of the feature.** Nine of our twenty-six ANE points came from a loader
call that runs before the weights are read, invisible to every accelerator
counter. Re-tuning the feature would never have found it — reproducing their
*sequence* did.

## The meta-rule

Most of these rules exist because a number was *right* under one protocol and
*wrong* about the system — the microbench that won, the probe that had no power,
the harness that measured post-EOS luck, the verdict on a stack missing its own
fix. The protocol is part of the claim. That is why `results/` ships raw
records and `harness/` ships the exact scripts: disagreement should be checkable
at the protocol level, not argued at the summary level.
