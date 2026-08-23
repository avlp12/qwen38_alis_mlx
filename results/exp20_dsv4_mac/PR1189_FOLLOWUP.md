Following up on item 3 (pool rows never rotated) — I've now patched and
validated it, and it was indeed the cause of the residual long-range slips.

The fix, mirroring the reference Compressor: after the RMSNorm, rotate the
last `rope_head_dim` dims of each pool row with the compress-theta + YaRN
rope at the row's block-start position (`i * ratio`). Since the positions are
strided rather than contiguous, I computed the angles directly from the
rope's `inv_freq` (same interleaved pairing as the main Q/K rope):

```python
pos = (base_row + mx.arange(P)) * ratio          # block-start positions
theta = pos[:, None] * rope.inv_freq[None, :]
# rotate ckv[..., -rope_head_dim:] with cos/sin of theta (interleaved pairs)
```

(Equivalently, a rope instance with its frequencies pre-scaled by `ratio` can
be applied with a contiguous row offset — same math.) Using the pool-row
count as `base_row` also keeps positions correct across chunked prefill. I
additionally replaced the all-zeros pool mask on the non-kernel prefill path
with the per-query clamp from item 2.

Result on the same 19K-token document task where I previously saw occasional
CJK token slips after fixing items 1–2: zero CJK slips over 250 generated
tokens, coherent and on-topic (4.9K unchanged and clean; single-box prefill
throughput unaffected, 422–473 tok/s on my M3 Ultra). So the three items
together fully account for the degradation I could measure on this branch.
