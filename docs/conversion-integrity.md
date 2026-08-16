# Conversion integrity — how a multimodal model becomes text-only without a warning

Qwen3.8-27B is a vision-language checkpoint (`Qwen3_5ForConditionalGeneration`,
`model_type: qwen3_5`): 27.78B parameters, of which 333 tensors / 0.461B params /
0.92 GB bf16 are the vision tower (`model.visual.*`), plus a 31-tensor MTP head.
This document is about two silent failure modes in the standard conversion path —
one that deletes a modality, one that corrupts the language model — and the design
that fixes both. The generalized, model-agnostic version of these rules lives in
[alis-dwq docs/PORTING_INTEGRITY.md](https://github.com/avlp12/alis-dwq/blob/main/docs/PORTING_INTEGRITY.md);
this is the Qwen3.8-specific record.

## 1. The converter drops the vision tower, and the artifact cannot tell you

mlx-lm's `qwen3_5` model class is text-only. Its `sanitize` returns only the
tensors the text model consumes, so every `model.visual.*` tensor is discarded —
and `utils.save_config` does `config.pop("vision_config")` on top. Together those
two lines convert a 27B VL model into a text-only model with no warning, and,
because the config key is popped as well, **the output is internally consistent**:
a stripped VL conversion is indistinguishable from a conversion of a genuinely
text-only model. Nothing in the artifact can flag the loss; only a comparison
against the source can.

The consequence in the wild, measured on 2026-08-15 by downloading the remote
weight index of the public MLX builds of this model found by a Hub search: **all
12 builds surveyed carried 0 vision tensors**, including builds with `-vision` in
the repository name. Seven of the twelve shipped a `preprocessor_config.json` —
the image pipeline's config — next to weights that cannot process an image, which
makes them look multimodal to tooling and fail on the first image. (One survey
lesson en route: a repo *name* is not evidence — the `-vision` repo measured 0
visual tensors — and neither is a repo's `index.json`, which can be a stale file
inherited from the source repo; the only trustworthy witness is the shard header
or the weight index actually served for that revision.)

Two scope corrections to that survey, in fairness and because they are checkable:

- The survey was of **text-stack (mlx-lm-lineage) conversions**. mlx-vlm-based
  conversions under `mlx-community/Qwen3.8-27B-*` (created 2026-08-14, missed by
  the survey's search) do preserve the 333 vision tensors — mlx-vlm is a VLM
  converter, so its pipeline never had this failure mode. Those builds drop the
  **MTP head** instead (0 `mtp.*` tensors against the checkpoint's 31).
- As of this writing, the builds in this campaign's HF collection are, to my
  knowledge, the only ones that preserve **both** subsystems — 333 vision tensors
  *and* the 31-tensor MTP head — alongside a quantized language model.

## 2. The fix: pass-through in `save()`, not a model patch

The tempting fix is to special-case `model.visual.*` inside `qwen3_5.py`. I put
the mechanism in `utils.save()` instead, so `convert` and `awq` pass through one
shared gate and any future architecture inherits it: a model class may declare
`passthrough_patterns` (here: `("model.visual.", "vision_tower.", "mtp.")`), and
tensors matching them that the model itself did not consume are copied to the
output **as original bytes** — no dtype cast, no quantization, no re-encoding.

Four design points, each of which was load-bearing in practice:

1. **Skip already-written keys with a suffix-aware comparison.** `sanitize`
   *reparents* modules — in this checkpoint `mtp.x` becomes
   `language_model.mtp.x` — so an exact-name check sees the reparented tensor as
   absent, re-emits it under its old name, and the index ends up describing one
   weight twice.
2. **Read the bytes back and verify before advertising them in the index.** An
   index entry pointing at an unwritten or truncated tensor fails at load time
   and looks exactly like a corrupt download.
3. **Keep `vision_config` only when the weights actually survived.** Both
   directions are bugs: advertising a tower a stripped checkpoint does not have,
   and hiding a tower that is present. The config must follow the bytes, decided
   at save time from what was written — not from what the source config said.
4. **Carry the preprocessor config with the weights.** The seven
   config-without-weights repos above are what the alternative looks like.

What it bought, verified: the tower survives **byte-exact, 333/333**, across the
8-bit / 6-bit / 4-bit builds; text inference is unchanged (decode 37.39 vs
37.61 tok/s, inside noise). And because mlx-vlm 0.6.13 already supports
`qwen3_5` — its `sanitize_key` rewrites the `model.visual.` prefix to
`vision_tower.` — the preserved checkpoints load for image work with **zero lines
of porting**. The ecosystem was never missing a vision implementation; it was
missing weights, because the converter threw them away quietly. Functional check
(not a VQA eval): a hand-drawn shapes image — red circle top-left, blue square
top-right, green triangle bottom — described correctly in color, shape, and
position.

Vision quantization policy: the tower stays bf16 at every tier. It is 0.92 GB =
1.66% of the model, so quantizing it saves almost nothing while adding an
unmeasured error term to the only visual path — and the upstream reference
practice agrees (`mlx-community/Qwen3-VL-8B-Instruct-4bit` measures text 4-bit,
vision all-bf16 by shard header).

## 3. The second trap, one function away: MTP presence as a format discriminator

Stock `qwen3_5.sanitize` decides whether to apply the one-time norm-weight shift
(the checkpoint stores RMSNorm gains as `gamma - 1`) with:

```python
should_shift_norm_weights = has_mtp_weights or has_unsanitized_conv1d
```

The intent of both disjuncts is "this is a raw HF checkpoint". But an
**already-converted** checkpoint that *kept* its MTP head still satisfies the
first disjunct, so its norms are shifted a **second** time (gamma 0.944 → 1.944).
Nothing crashes. Generation collapses quietly: on my Korean probe, NLL
1.679 → **17.460** — worse than a uniform distribution over the 248,320-token
vocabulary.

And the failure frames the wrong suspect: builds *without* the MTP head measure
perfectly, so the evidence reads as "the MTP-preserving build is broken" — the
bug incriminates precisely the feature that exposes it.

### The discriminator principle

**A format discriminator must be something the transformation itself destroys.**
The second disjunct, `has_unsanitized_conv1d`, is a correct witness: the raw
Conv1D layout exists only before conversion, and conversion consumes it, so it
cannot survive into the converted artifact. The MTP head is *incidental* — it
passes through conversion untouched and is therefore evidence of nothing. Stated
as an invariant: `sanitize` must be idempotent, and a discriminator that survives
its own transformation breaks idempotence by construction. Before shipping any
"is this raw?" test, ask what the second application does; if the answer is
"shifts it again", the witness is wrong.

The fork's fix is one line — `should_shift_norm_weights =
has_unsanitized_conv1d` (see `code/patches/fork-vs-upstream.diff`). Upstream had
the same class of bug reported from a 35B MoE
([issue #1197](https://github.com/ml-explore/mlx-lm/issues/1197),
[PR #1623](https://github.com/ml-explore/mlx-lm/pull/1623)); I reproduced it
independently on this dense 27B and filed
[**PR #1735**](https://github.com/ml-explore/mlx-lm/pull/1735), leaving the
validation data on the threads.

A corollary worth internalizing: a silent-collapse bug can sit in a popular
converter for a long time, because the models it breaks are the unusual ones
nobody re-measures — here, the first build in the ecosystem to *keep* the head
was the first one able to trigger it.

## 4. The incident behind the incident: which library did the harness import?

The double shift was found while chasing an apparent quality collapse — and the
collapse was being measured with **stock mlx-lm**, because the fork and the
installed release share an import name and the fork was not first on `sys.path`.
Every measurement predating the fix had to be re-run. The harness now pins the
fork path explicitly and **fails hard** if `mlx_lm` resolves from
`site-packages`, printing the resolved path on every run (see
`harness/measure.py`, whose header comment is the incident report in code form).
`harness/table2.py` additionally carries a tripwire: any build whose Korean probe
NLL exceeds the bf16 reference's by more than 3x triggers a warning to check for
a stock import — the bug's signature, encoded as a guard.

## Ledger references

Survey and passthrough: `VISION_FEASIBILITY` nodes (summarized in the
[alis-dwq case study](https://github.com/avlp12/alis-dwq/tree/main/examples/qwen3.8-27b));
double shift and harness pin: the case study's Finding 2; converter follow-up
(AWQ path leaves MTP bf16): [LEDGER.md](LEDGER.md) `[PA18]`, with the interim
tool shipped as `harness/quantize_mtp.py`.
