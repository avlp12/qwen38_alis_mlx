# results/ — raw measurement records

Small JSON only — no weights, no `.npy` arrays, no raw logs. Every summary
number in the README and `docs/` resolves either to a ledger node
([docs/LEDGER.md](../docs/LEDGER.md)) or to a file here; the scripts that
produced these files are in [harness/](../harness/). Builds are named by the
campaign's local convention: `q{4,6,8}` = first uniform builds (pre
vision/MTP passthrough), `q{4,6,8}v` = vision+MTP-preserving uniform builds
(`v` = vision; these are the published uniform tiers), `q4awq`/`awq2`/`awq3` =
AWQ recipe iterations (3 = final), `q4awq3m` = AWQ with the MTP head quantized
to 4-bit (the shipped 4-bit), `bf16` = the unquantized source.

## Per-build measurement records (`m_*.json`)

Produced by `harness/measure.py`; rendered as a table by `harness/table2.py`.
Each records build size, bits/group, load+probe peak memory, per-prompt decode
tok/s, 2048-token prefill tok/s, and a per-slice probe (NLL plus the full top-1
id sequence, so agreement is recomputable without re-running the model).

| file | build |
|---|---|
| `m_bf16.json` | bf16 source — the reference all top-1 agreement is computed against |
| `m_q4.json` / `m_q6.json` / `m_q8.json` | first uniform 4/6/8-bit builds (no vision, no MTP — pre-passthrough) |
| `m_q4v.json` / `m_q6v.json` / `m_q8v.json` | vision+MTP-preserving uniform builds (published 6/8-bit tiers; q4v later replaced by AWQ) |
| `m_q4v_retry.json` | re-measurement run of q4v (protocol repeat) |
| `m_q4awq.json` / `m_q4awq2.json` / `m_q4awq3.json` | AWQ 4-bit recipe iterations; `awq3` is the final recipe |
| `m_q6awq.json` / `m_q6awq2.json` / `m_q6awq3.json` | AWQ at 6-bit (evaluated; not shipped — uniform kept at this tier) |
| `m_q8awq3.json` | AWQ at 8-bit (evaluated; not shipped — indistinguishable from uniform and bf16) |

## Quality — corpus-scale strided perplexity

| file | what it is |
|---|---|
| `ppl_out/ppl_{bf16,q4v,q6v,q8v,q4awq3,q6awq3,q8awq3}.json` | per-slice strided-PPL summaries (window 2048, stride 512; en = wikitext-2 test, ko = Korean Wikipedia, code = CPython stdlib) from `harness/ppl_eval.py` |
| `ppl_verdict.json` | the paired verdict across all seven: absolute NLL/PPL per slice, excess vs bf16 with 512-token block SE, AWQ-vs-uniform deltas, and tier gaps — rendered by `harness/table3.py` |

The per-token NLL arrays (`.npy`) behind these summaries are not shipped
(size); `harness/ppl_eval.py` regenerates them, and
`harness/build_eval_corpus.py` rebuilds the exact corpus slices.

## KV cache and drafter-tap diagnostics

| file | what it is |
|---|---|
| `kv_q4v.json` / `kv_q8v.json` | KV-cache quantization sweep (`kv_bits` in {none, 8, 4}) at 16K context on the 4-bit and 8-bit builds: prefill, decode, peak, top-1 ids per setting (`harness/kv_measure.py`) |
| `tap_drift.json` | drift of the target's tap-layer hidden states (drafter inputs) per quant recipe vs bf16: cosine, relative L2, RMS ratio per tap layer, plus logit KL (`harness/tap_drift.py`) — the data behind rejecting the scale-drift hypothesis |

## Speculative decoding (`out2/`)

| file | what it is |
|---|---|
| `out2/ref_bf16.json` | bf16 greedy reference streams — the common token sequences all paired acceptance probes score against |
| `out2/bench_{q4v,q4awq3,q4awq3m}.json` | plain vs MTP bench records per build (fixed 240-token windows, pre-EOS-audit protocol; superseded for cross-build claims) |
| `out2/bench3_{q4v,q4awq3,q4awq3m}.json` | the 4-prompt plain / MTP k=2 / DSpark records behind the headline table, with build fingerprint (fast_qmm state, MTP head class/bits) and losslessness checks |
| `out2/dspark_{q4v,q4awq3,q4awq3m}.json` | DSpark-path records per build (acceptance, tok/s, prefill, peak) |
| `out2/probe.json` / `out2/probe_dense.json` | paired acceptance probes on the common bf16 streams (strided / dense per-position with draft dumps) — the AWQ-vs-uniform acceptance comparison done right |
| `out2/acceos_{q4v,q4awq3}.json` | EOS-cut paired acceptance per workload — the corrected protocol after the contamination incident |
| `out2/eos_q4v.json` | the EOS diagnostic itself: where answers actually end inside the fixed measurement window, per configuration |
| `out2/prefill.json` / `out2/prefill_68.json` | 2048-token prefill spot-checks across the 4-bit builds / the 6- and 8-bit builds |

## Speculative restatement + real-world sampling (`spec_restate/`)

The records behind the 2026-08-16 retroactive restatement of the speculative
headline (`[J7]`/`[I78]` in the ledger): EOS-cut protocol, long-form 4-prompt
set (chat/code/math/ko), medians of 3 for sampled rows. Scripts:
`harness/sweep_b.py` (gate/depth sweep), `harness/bench_sampling.py`
(real-world sampling + greedy regression), `harness/bench_d.py` (8-bit target),
`harness/calib_regress.py` (rejection-sampling losslessness calibration).

| file | what it is |
|---|---|
| `spec_restate/greedy_regress.json` | the canonical greedy table on the 4-bit build: plain 37.6 / DSpark 48.3 / MTP k=2 46.8 / gated MTP k=4 52.8 |
| `spec_restate/samp_240.json` / `samp_1024.json` | first real-world measurement — Qwen shipped defaults (temp 1.0 · top-p 0.95 · top-k 20), truncated rejection sampling, 3 reps per cell |
| `spec_restate/gate_240.json` / `gate_1024.json` | the `min_draft_p` gate sweep that found the winning lever (`[I72]`) |
| `spec_restate/sweep_1024.json` | the ungated depth/rejection sweep (bare k monotonically loses past 2, `[I71]`) |
| `spec_restate/bench_d_q8v.json` | the 8-bit-target reproduction: plain 21.8, MTP k=2 1.42x, DSpark 1.54x at block 4 (`[I73]`) |
| `spec_restate/calib_regress.json` | truncated rejection-sampling validation: synthetic-oracle TV, greedy-limit equality, defer/sync bitwise checks (`[I76]`) |

## Two-box prefill (`bench_2box/`)

| file | what it is |
|---|---|
| `bench_2box/results_2box.json` | the full 1-box vs 2-box sweep (N in {2048, 8192, 32768} x chunk sizes, with per-chunk stage accounting) — rendered by `harness/analyze_2box.py` |
| `bench_2box/bonus32k.json` | the 32K chunk-size sweep that located the shared 1024 optimum |
| `bench_2box/verify_2box.json` | the bitwise correctness record: per-layer cache comparison, logit max-abs, greedy continuation |
| `bench_2box/serving_verdict.json` | the `mlx_lm.server --prefill-2box` integration verification: no-regression, correctness, TTFT A/B (20.3 → 11.9 s), multi-turn incremental, runner-death policy, and the bugs found |
