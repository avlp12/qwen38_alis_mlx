# PR#1189 코멘트 초안 v2 (3-렌즈 적대검증 반영 — 게시 승인 대기)

Found three deviations from the reference implementation while testing this
branch on long inputs. Sharing with fixes for the first two, since together
they explain the degraded generation some users may be seeing on longer
prompts.

**1. rope/YaRN assignment.** The reference `Attention.__init__` builds one
rope per layer:

```python
if self.compress_ratio:
    original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
else:
    # disable YaRN and use base rope_theta in pure sliding-window attention
    original_seq_len, rope_theta = 0, args.rope_theta
```

i.e. compressed layers (41 of 43) rotate main Q/K with `compress_rope_theta`
(160000) + YaRN, and the ratio-0 sliding-window layers (0, 1, 42) use base
theta with YaRN disabled. This branch applies `rope_theta` (10000) + YaRN to
main Q/K on every layer — the `self.compress_rope` instance is created but
never called. Empirically, on my M3 Ultra with the 4-bit community quant,
real-document prompts are fine at 1.5K, marginal at 2.2K, and fully broken at
4.9K ("? 2:2:2:..." style output), while the same prompts through llama.cpp's
deepseek4 arch stay coherent — which is what localized it to this branch.
(Other deviations below plausibly contribute past ~2K as well, e.g.
`index_topk=512 × ratio 4` puts the indexer's activation threshold right at
2048, so I'm attributing by differential measurement rather than claiming a
single cause.)

Fix (one conditional):

```python
self.rope = DeepseekV4RoPE(
    self.rope_head_dim,
    args.compress_rope_theta if self.compress_ratio else args.rope_theta,
    args.rope_scaling if self.compress_ratio else None,
)
```

After this, greedy generation over 4.9K- and 19K-token documents is coherent
and on-topic (occasional CJK token slips remain at 19K — see item 3).

**2. Compressed-pool visibility during prefill.** The prefill path prepends
the compressed pool with an all-zeros additive mask, so every query in a chunk
can attend to pool rows that summarize positions *after* it. All three
implementations I checked (the reference `model.py`, HF transformers'
`modeling_deepseek_v4`, and FreeToken's port) clamp per query: row `i` is
visible to position `p` iff `i < (p + 1) // ratio` (the pool row's last source
token must be ≤ p). Besides the mid-chunk state corruption this causes, it
also makes prefill and decode inconsistent (the decode path only ever sees
completed pool rows, so teacher-forced evals of the prefill path read
optimistic). I verified with a fused kernel that implements the reference
visibility — outputs become clean once both (1) and (2) are fixed.

**3. Pool rows are never rotated.** The reference applies compress-theta rope
to each pool row at its block-start position before it enters the K/V stream
(`freqs_cis[:cutoff:ratio]` in the reference Compressor; same in transformers
and FreeToken), so query·pool-row dot products encode relative distance. This
branch's Compressor returns the normed rows without any rope — which looks
like what `self.compress_rope` was created for. I haven't patched this one
yet; it plausibly accounts for the residual long-range slips I still see at
19K after fixing (1) and (2).

There are a few further indexer-path deviations (per-query masked top-k vs a
single shared top-k over the mean, relu+linear weighting vs sigmoid, and the
S=1 decode path bypassing top-k entirely so the pool is fully attended past
`index_topk × ratio` tokens) — happy to detail those, share benchmarks, or
test other configurations.
