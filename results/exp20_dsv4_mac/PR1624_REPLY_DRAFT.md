Thanks for the review! Two points, and I'm happy to rework the PR either way.

**On the identity.** You're right that "exactly" was sloppy wording for what a
float kernel does — but the identity itself is plain algebra, not an
approximation:

```
sqrt(mean(x²) + ε/D) = sqrt((sum(x²) + ε)/D) = sqrt(sum(x²) + ε) / sqrt(D)
```

so `rms_norm(x, None, ε/D) = sqrt(D) · x / sqrt(sum(x²) + ε)` holds in real
arithmetic (all three `fast.rms_norm` backends add eps to the mean inside the
rsqrt), with the `sqrt(D)` absorbed by the `inv_scale` powers already in the
code. The ~1e-7 residual in my table is fp32 rounding noise, not a semantic
gap. For completeness I also checked FLA's chunked path — `fla/modules/
l2norm.py` uses the same `1/sqrt(sum(x²) + 1e-6)` form as the fused-recurrent
kernel, so both prefill and decode reference paths agree on the semantics.

**On reverting to the pre-#853 `_l2norm`.** I'd gently push back on that one:
it adds eps *outside* the sqrt — `x / (‖x‖ + 1e-6)` — which is a third
semantics, also different from the FLA kernels (`x / sqrt(‖x‖² + 1e-6)`, eps
inside on the sum). As ‖x‖ → 0 its denominator tends to 1e-6 instead of the
reference's 1e-3, so near-zero vectors get amplified ~1000×. Measured against
the FLA form (fp32, D=128, max|Δ|/max|ref|, order-of-magnitude stable across
seeds):

| amplitude | this PR (ε/D) | pre-#853 `x/(‖x‖+ε)` |
|---:|---:|---:|
| 1.0 | ~1e-07 | ~2e-07 |
| 1e-2 | ~1e-07 | ~4e-05 |
| 1e-4 | ~1e-07 | ~0.4 |
| 1e-6 | ~1e-07 | ~70× |

So the revert would trade the current mismatch for a different (larger)
small-magnitude one.

If the `ε/D` trick reads too clever, the reference can also be implemented
literally — note the scale placement has to change with it, since the
explicit l2norm doesn't carry `rms_norm`'s internal `sqrt(D)` factor:

```python
q = self.scale * q * mx.rsqrt((q * q).sum(-1, keepdims=True) + 1e-6)
k = k * mx.rsqrt((k * k).sum(-1, keepdims=True) + 1e-6)
```

(i.e. the pre-#853 scale structure — `scale` once on q, none on k — with eps
moved inside the sqrt.) That matches the reference the same way this PR does;
the one thing the `fast.rms_norm` form keeps is fp32 accumulation on bf16
inputs. Happy to update the PR to either form — just say the word.
