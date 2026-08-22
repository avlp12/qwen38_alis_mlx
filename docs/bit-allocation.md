<title>Bit allocation</title>

# What a competitor's public quant told us, and which half of it transferred

Unsloth's Dynamic 3.0 quants landed on the same model we had just spent a
campaign quantizing. Their documentation describes the method in four sentences —
a better imatrix calibration set, "improved layer selection", "many more
quantization techniques", no QAT — and discloses no allocation rule. So we read
the artifacts instead, and then measured what we found against our own builds.

Two things came out of it. The list of tensors they protect transferred
and is worth **16% of our published 4-bit build's KL**. Their allocation
*algorithm*, as we reconstructed it from their imatrix, did not: every version we
built lost to plain uniform 4-bit, and the reasons why are the more useful half
of this document.

## Reading a quantization scheme without downloading it

Two artifacts are public and neither needs a large download.

**The imatrix** (`imatrix_unsloth.gguf`, 13.6 MB) is a GGUF whose tensors are
`<name>.in_sum2` — per-input-channel sums of squared activations — plus
`.counts`. Its header says `chunk_count 1251 × chunk_size 8192`, so it is
**10,248,192 tokens** of calibration. For scale, our own AWQ pass used
`--num-samples 128 --sequence-length 512` = **65,536 tokens**, 156x less.

**The allocation** does not need the weights at all. GGUF puts its full tensor
table — name, shape, type, offset — at the front of the file, so a few HTTP range
requests read the entire bit assignment of a 14 GB quant. We pulled four budget
points (2.86 / 3.84 / 4.17 / 7.41 bpw) and diffed them; how a scheme *degrades*
under a shrinking budget is exactly its priority order, stated in public.

What that reads out:

- Allocation is genuinely **per tensor**, not per tensor-class: `ffn_down` alone
  spans eight types from IQ2_S to Q6_K across its 65 layers.
- **Budget-invariant**: `ssm_alpha`/`ssm_beta` sit at Q8_0 in every build,
  including the 2.86 bpw one; the MTP projection stays Q6_K throughout.
- **Attention is asymmetric**: mean bpw v 5.57 > k 5.12 > o 4.70 > q 4.00. At
  2.86 bpw, `attn_v` still holds Q4_K/Q5_K while `attn_q` falls to Q2_K/IQ2_XXS.
- **Input and output are asymmetric**: `token_embd` is always the lowest tensor
  in the file; `output` is always well above the budget.
- **FFN rises with depth** (layer-to-bpw correlation 0.65-0.75).

## The two halves

Correlating their allocation against their own imatrix separates the design
cleanly. For FFN it explains the trend: activation energy grows four orders of
magnitude from layer 0 to 64 (`ffn_down` log10 0.19 → 4.63), and their bit
assignment tracks it at r = 0.65-0.78. For attention it explains nothing —
`attn_v` is high regardless of energy.

The second half is not a sensitivity result, it is an **arithmetic** one. Under
GQA, `attn_k` and `attn_v` are 0.33% of the parameters *each*; `ssm_alpha` and
`ssm_beta` are 0.04% each. Protecting all four costs 0.74% of the file. They are
not high because they are precious — they are high because they are almost free.

That framing predicts which half is portable, and the prediction held.

## What we measured

Same 27B checkpoint, MLX affine, group size 64, exact full-vocab KL against bf16
over non-overlapping 2048-token windows. All comparisons below are **paired** —
reference and every target resident in one process, alternating per window, so
between-window variance cancels and the standard error drops about fourfold.

Against uniform 4-bit:

| arm | GiB | en | ko | code |
|---|---:|---|---|---|
| **B** — k, v, α, β → 8-bit | 15.27 | −7.0% (t=−8.7) | −5.2% (t=−14.5) | −4.8% (t=−9.3) |
| **F** — B + embed 4→3 + head 4→5 (**byte-identical to B**) | 15.27 | −16.3% (t=−14.6) | −13.1% (t=−15.6) | −3.8% (t=−5.8) |
| **H** — B + head 4→5, no demotion | 15.42 | −17.5% (t=−18.4) | −15.9% (t=−30.3) | −5.9% (t=−10.9) |
| C — imatrix knapsack, raw weighting | 15.17 | +15.8% | +23.7% | +37.2% |
| C2 — imatrix knapsack, relative weighting | 15.17 | +3.6% | +2.8% | +2.0% |

And on top of AWQ, against our published 4-bit build:

| | en | ko | code |
|---|---|---|---|
| AWQ + B's promotions + head 5-bit | **−16.6% (t=−30.8)** | **−16.4% (t=−37.1)** | **−3.9% (t=−7.1)** |

KL 0.03505 → 0.02924 on en; against the uniform baseline that is −33.6%. The
promotion and AWQ **compose**: they attack different parts of the error, so
unlike the accelerator/pipeline pair we measured earlier, the gains multiply
rather than compete.

### Is it the bytes or the tensors?

The obvious objection to any promotion result is that we simply spent more. So we
built the control: the *same* 248 MiB, on 23 arbitrary mid-depth `mlp.gate_proj`
tensors instead of the chosen five, byte-matched to within 0.02%.

| vs H, paired | en | ko | code |
|---|---|---|---|
| K — same bytes, arbitrary tensors | **+19.3% (t=+13.6)** | **+15.9% (t=+21.3)** | +1.8% (t=+2.3) |

K's absolute KL moved 1.2% off the uniform baseline. The bytes bought essentially
nothing; the *choice* is worth 16-19%.

And the fixed-size question, asked cleanly: J promotes the head to 6 bits and pays
for it by dropping `embed_tokens` to 3, byte-identical to H.

| vs H, paired | en | ko | code |
|---|---|---|---|
| J — same bytes, redistribution | −1.5% (t=−2.2) | −1.0% (t=−1.4) | +1.9% (t=+3.7) |

A wash, with a prose/code trade. Which retires the earlier F-beats-B result as
evidence about redistribution: F did not win because it redistributed, it won
because it touched the head at all, and B did not. Against an arm that has already
promoted the head, redistribution has nothing left to add.

Two more calibrations worth stating. Storing scales and biases as F16 instead of
BF16 is free in bytes and worth −0.2 to −0.8% (significant only on ko, t=−6.3) —
small, but it is the cheapest thing on this list. And the head does not stop
paying at 5 bits: 6 bits buys another −2.9% / −4.1% / −0.4% (t=−15.6 / −21.4 /
−3.1) for a further 1% of size.

## What we got wrong on the way

We first concluded that "the gain is in cheap promotions, not in redistribution,"
on the strength of C and C2 losing at fixed size. Two checks killed it.

**F and B are byte-identical** — 16,394,118,624 bytes each — and F is exactly a
fixed-size redistribution: it demotes `embed_tokens` to pay for the head. It
beats B by −16.3% vs −7.0% on en. Our evidence for "demotion is a net loss" had
been H > F, and H is 0.97% *larger*; that comparison was not size-controlled.

**The knapsack was never allowed to make the winning move.** The published
imatrix has 496 entries, all `blk.*` — no `output.weight`, no `token_embd.weight`,
no MTP projection. Our allocator pinned those at 4 bits. Roughly two thirds of
H's gain is precisely `lm_head` 4→5. "Redistribution loses" was therefore never
tested; what we actually showed is narrower and still useful: **our imatrix
knapsack lost to uniform**, because its objective ignores cross-layer
amplification and its search space excluded the output head.

Given the head, the relative-error criterion gets it backwards — it demotes
`lm_head` to 3 bits. An error in the output head lands directly on the logits
with no downstream normalization to absorb it, which a per-tensor relative-error
score cannot see.

## What we changed

- `mlx_lm/quant/awq.py` takes `--bit-map` (a JSON of per-tensor widths applied at
  the final quantize) and `--lm-head-bits` (the output head separately from the
  input embedding), so an AWQ build can carry a graded allocation.
- Recommended 4-bit recipe for this architecture: run AWQ, then `k_proj`,
  `v_proj`, `in_proj_a`, `in_proj_b` → 8 bits and `lm_head` → 5-6 bits, with F16
  scales. Cost about 1.7% of size, worth **16% of KL on top of AWQ**. Demoting
  `embed_tokens` to fund the head is a real option — at matched bytes it is worth
  most of the gain — but it costs on code, so prefer paying the 1.7% if you can.

## Two traps worth carrying

**A surrogate that improves while the target degrades.** Our knapsack objective
moved 60.7% the right way and KL moved 16-37% the wrong way. A criterion is a
hypothesis; the only referee is the metric you actually ship on. We ran the
build rather than trusting the number, which is the only reason the raw-weighted
allocator did not end up in a published model.

**A conclusion that survives only because the control is the wrong size.** We
published "redistribution does not pay" off a comparison between two builds that
differed by 0.97% in bytes, while a byte-identical pair sitting in the same
results directory said the opposite. Before any claim of the form "X does not
help", check that the thing X was compared against is actually the same size —
and that the search which failed to find X was allowed to consider it.

**A per-tensor bit map that silently applies to nothing.** Our AWQ integration
looked up block-relative module paths (`self_attn.k_proj`) in a map keyed by
checkpoint names (`model.language_model.layers.3.self_attn.k_proj`). Not one
entry matched; `nn.quantize` fell through to the global width, the log printed a
count of map entries rather than of hits, and the written config recorded 4-bit
everywhere. An hour of compute produced a build we briefly read as evidence that
promotion does not compose with AWQ. It does — by 16%. Count the *hits*, not the
inputs, and verify the predicate before the build, not after.

**A hardcoded path that resolves to something else on the other machine.** The
same fork (file hashes equal), the same source (config and index sha256 equal),
the same MLX and Python produced, on our second box, a checkpoint with the MTP
module unquantized and the vision tower missing entirely — 8 fewer quantized
modules and 333 dropped tensors, no warning anywhere. Every environment probe we
ran said the two machines were identical, because every probe passed the fork
path explicitly.

The cause was one line of ours: `sys.path.insert(0, "~/glm5.2/mlx-lm")`. On the
build machine that is the current fork. On the other machine a stale tree from a
previous campaign still sits at that path — no MTP port, no
`passthrough_patterns` — and inserting it at position 0 shadowed the `PYTHONPATH`
that pointed at the right one. Three symptoms, one line: the missing MTP head
accounts for the module-count gap, the missing passthrough declaration for the
silent log, and `sanitize` dropping vision with nothing to preserve it for the
lost tower.

Two guards now, because size was the only thing that caught it: the fork comes
from `FORK` rather than a hardcoded path, the loaded tree is **asserted** to
declare `passthrough_patterns` before anything is built, and the finished
checkpoint is **counted** — a build whose index has zero vision or zero MTP
tensors fails instead of shipping. A probe that specifies the thing under test
cannot detect a fault in how that thing gets selected.

Raw records: `results/exp16_unsloth/`, including the adversarial review that
overturned our first conclusion. Ledger `[I173]`-`[I189]`, `[RA62]`-`[RA66]`,
`[CA36]`-`[CA39]`, `[PA58]`-`[PA59]`.
