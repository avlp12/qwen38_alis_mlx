# PR#1189 코멘트 초안 (게시 대기 — 승인 시 1인칭 그대로 게시)

Found a rope assignment bug while testing this branch on long inputs, plus a
mask-causality issue in the compressed-attention prefill path. Sharing both
with fixes since they explain the degraded generation some users may be seeing
past ~2K tokens.

**1. rope/YaRN assignment (generation breaks progressively past ~2K).**
The reference `Attention.__init__` builds one rope per layer:

```python
if self.compress_ratio:
    original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
else:
    # disable YaRN and use base rope_theta in pure sliding-window attention
    original_seq_len, rope_theta = 0, args.rope_theta
```

i.e. compressed layers rotate main Q/K with `compress_rope_theta` (160000) +
YaRN, and ratio-0 layers use base theta with YaRN disabled. This branch applies
`rope_theta` (10000) + YaRN to main Q/K on every layer. The phase error grows
with position: on my M3 Ultra with the 4-bit community quant, real-document
prompts are fine at 1.5K, marginal at 2.2K, and fully broken at 4.9K
("? 2:2:2:..." style output), while the same prompt through llama.cpp's
deepseek4 arch is coherent — which is what localized it to this branch.

Fix (one conditional):

```python
self.rope = DeepseekV4RoPE(
    self.rope_head_dim,
    args.compress_rope_theta if self.compress_ratio else args.rope_theta,
    args.rope_scaling if self.compress_ratio else None,
)
```

After this, greedy generation over 4.9K- and 19K-token documents is coherent
and on-topic (occasional CJK token slips remain at 19K — possibly the same
family as the earlier report on #1192).

**2. Compressed-pool visibility during prefill.** The prefill path prepends the
compressed pool with an all-zeros additive mask, so every query in a chunk can
attend to pool rows that summarize positions *after* it. The reference clamps
per query: row `i` is visible iff `i < (p + 1) // ratio`. This contributes
mid-chunk state corruption on top of (1); I verified with a fused kernel that
implements the reference visibility (outputs become clean once both are fixed).

Happy to share benchmarks or test other configurations.
