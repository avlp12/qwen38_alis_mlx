# oMLX 이슈 초안 (게시 승인 대기 — jundot/omlx, 1인칭 영문)

**제목**: Prompt priming never activates for legacy-MTP DeepSeek-V4 under BatchGenerator (two coupled activation bugs; +19% long-prompt decode once fixed)

---

While serving DeepSeek-V4-Flash-0731 (legacy single `mtp.0`, not DSpark) through
the BatchGenerator path, I noticed `primed=0` on every request and traced the
prompt-priming activation chain. Two independent bugs each kill it on their own;
both need fixing before priming can engage. With both fixed, on a 2.1K-token
prompt at fixed depth-3 chaining I measure draft acceptance d1 81.5% → 95.6%
(d2 54.5 → 66.7%), tokens/verify-cycle 2.37 → 2.81, decode +19.4% — so this is
worth reclaiming.

**Bug 1 — `_anchor` only accepts plain-int offsets, but batch caches never
expose one.** `prompt_priming._anchor` probes `type(c.offset) is int`. Under
BatchGenerator, caches are merged into `BatchKVCache` /
`BatchRotatingKVCache` at `PromptProcessingBatch.__init__` (mlx-lm
`generate.py` → `_merge_caches`), whose `.offset` is a 1-element `mx.array`
**even for a single request (B==1)**. So `_anchor` returns None and
`maybe_capture` bails silently on every capture site — the last (hidden,
next-token) pair is never folded and `take_primed` later discards the seam on
offset mismatch. Notably `_activation_offset` in the same file already
tolerates 1-element array offsets ("activation-time only, never per-chunk"),
so I believe the capture-time scalar-cache assumption is unintentional.

Fix I'm running: an anchor view that unwraps size-1 array offsets to int (the
`(1, S)` input guard already restricts capture to singleton timelines, so B>1
ambiguity doesn't arise). Two caveats worth deciding consciously:
- this adds a per-chunk `int()` sync, which contradicts the "never per-chunk"
  comment — reading `BatchRotatingKVCache._offset` (a python int) would be a
  sync-free alternative, but note `_offset` is the buffer length, not the
  token count, so it needs care;
- with batch capture alive, a B>1 prefill window is invisible to capture; a
  cancelled request's stale ctx could in principle seam-match a later chunk
  boundary (2048-aligned). Output correctness is protected by verify either
  way, but dropping the ctx on any `inputs.shape[0] != 1` forward would keep
  the "never a wrong history" invariant.

**Bug 2 — `mtp_take_primed` hook is DSpark-only but registered
unconditionally, and the generic seam is unreachable behind it.** The
deepseek_v4 patch registers `mtp_take_primed` on the class; for non-DSpark
(legacy MTP) it returns None unconditionally. `prompt_priming.take_primed`
returns whatever a callable hook returns, so the generic ctx seam below it can
never run for legacy models — even with Bug 1 fixed, activation still dies.

Fix I'm running: treat a hook returning None as "declined ownership" and fall
through to the generic seam. I checked the other hook implementations for
safety: DSpark and inkling both pop their own ctx before returning None, so
fallthrough cannot pick up a foreign ctx; and on DSpark-enabled models the
patched `__call__` returns before generic capture, so there is no generic ctx
to pick up. (A defensive `isinstance(_PrimeCtx)` check in the generic seam
would future-proof this.)

**Related observation (no kernel bug).** The same scalar-cache assumption also
silently disables the `wsdpa_prefill` fast path under batch caches (its int
offset guard falls back to stock SDPA), so batch serving quietly loses that
optimization too. I initially suspected the kernel itself of producing wrong
output in the MTP (`_standard_mask`, no pool) context, but differential
testing against stock SDPA across cold / rotated / cache-None shapes (H=64,
L up to 2048, RotatingKVCache with real trim/rotation) shows bf16-noise-level
agreement everywhere — the kernel is fine; I'm happy to share the diff
harness. The systemic pattern is just: pre-batch-era int-offset probes die
silently in the batch engine, and priming happened to be the user-visible
casualty.

Happy to send these as a PR (anchor view + hook fallthrough + the defensive
guards) if that's welcome.
