# Two-box prefill — a layer-pipelined split over Thunderbolt 5, bitwise-exact, 440 to 755 tok/s

Single-box prefill on this model closed at 92.7-96.3% of the engine ceiling
([kernels.md](kernels.md), `[PA19]`) — the remaining levers summed to under 7%,
so the next factor had to come from a second machine. This is the design and the
measured result of splitting prefill across two M3 Ultras over a direct
Thunderbolt 5 link: **8192-token prefill 427 → 733.5 tok/s (1.72x), 32K
388 → 733.6 (1.89x), with bitwise-identical outputs.** Ledger: `[I55]`-`[I63]`,
`[PA21]`-`[PA22]` in [LEDGER.md](LEDGER.md); raw records in
`results/bench_2box/`.
>
> **Re-measured on mlx 0.32.1 (2026-08-20).** An earlier version of this note divided
> the new single-box baseline (441.3) by the *old* two-box figure and concluded the
> speedup had fallen to 1.66x. That was wrong: the pipeline moved too. Re-running the
> canonical harness end to end on 0.32.1 — bitwise verification re-passed first — both
> arms gain about 3% and the ratio is unchanged.
>
> | N | old 1box | old 2box | old ratio | new 1box | new 2box | **new ratio** |
> |---:|---:|---:|---:|---:|---:|---:|
> | 2048 | 429.3 | 625.7 | 1.457x | 442.4 | 646.1 | **1.460x** |
> | 8192 | 427.3 | 733.5 | 1.717x | 440.2 | **755.2** | **1.715x** |
> | 32768 | 387.6 | 712.5 | 1.838x | 398.4 | 729.7 | **1.831x** |
>
> Best chunk per cell, three reps, alternated, cooldowns. The 32K rows are chunk-2048
> like-for-like; the chunk-1024 bonus point that produced the headline 1.89x is being
> re-measured separately. Raw: `results/exp13_2box321/`.

## 1. Feasibility gates, measured before any code

Four numbers had to clear before the build was worth starting `[I55]`:

- TB5 TCP sustained: **4.35 GB/s**; a 16 MiB slab one-way in 3.0 ms.
- Per-layer compute at prefill: ≈197 ms per chunk-stage — a hiding margin of
  ≈60x over the boundary transfer.
- GDN boundary state: **3.15 MB/layer** (`[1, 48, 128, 128]` fp32, `[I58]` —
  an early brief's "6.3 MB" was a misread of a dataclass default belonging to a
  different model).
- Cache retrieval arithmetic checks out exactly: at 2048 tokens, 144,080,896
  bytes = 24 x (ssm 3,145,728 + conv 61,440) + 8 x 2 x KV.

## 2. Design

`mlx_lm/prefill_2box/` (snapshot in `code/prefill_2box/`, opt-in API
`TwoBoxPrefill`, existing paths untouched) `[I56]`:

- **Layer-major split**: box A (epsilon) runs layers 0-31 + embedding; box B
  (gesicht) runs layers 32-63 + norm + head. Decode stays on box B alone.
- The boundary activation `[1, T, 5120]` bf16 crosses as a u16 bit-view over raw
  TCP on the TB5 link — verified bit-lossless round-trip.
- A server-side send thread and a client-side receive thread give three-way
  overlap: A computes chunk i+1 while the link carries chunk i while B computes
  chunk i-1.
- `stream_kv` (opt-in): pre-send each chunk's KV slab over the idle link —
  pipeline-neutral (+0.01 s at 8192) but cache install drops 0.256 → 0.067 s
  (8192) and 0.474 → 0.144 s (32K), worth −0.18 s TTFT `[I62]`.

## 3. Correctness — strict bitwise, with stated preconditions

Against a single box running the identical chunk schedule: all 64 layers' cache
tensors **128/128 bit-identical**, final logits max abs 0.0, greedy 48-token
continuation identical `[I57]`. Preconditions that make bitwise possible: same
chip (M3 Ultra), same OS build, same MLX build on both boxes (the campaign
pinned a dev wheel on each). For context, chunked-vs-single-shot prefill on
*one* box differs by up to 0.25 max abs logits — normal reduction-order drift
from chunking itself, unrelated to the split. The serving A/B later reproduced
this at the HTTP level: with schedule-matched `--prefill-step-size 1024` on both
sides, 28/28 exact-equality checks passed (`results/bench_2box/serving_verdict.json`).

## 4. The bubble law

For a two-stage layer split, total time is

```
T = (C + c_max) / 2
```

where C is the same-schedule single-box time and c_max the largest chunk's time
`[I59]`. The bubble is half the largest chunk and is the same wherever that
chunk sits in the schedule — so **uniform chunks are optimal** and tail-shrinking
tricks buy nothing. Measured within 1.5-4% of the model everywhere (residual =
per-chunk Python/link fixed cost).

## 5. Results

Crossed order, cooldown-controlled, twice-reproduced (repeat spread <= 0.15%);
tok/s = N / t_prefill `[I60]`:

| N | 1-box (best chunk) | 2-box (best chunk) | prefill | TTFT |
|---|---|---|---|---|
| 2048 | 4.768 s (429) | 3.272 s @512 (**626**) | 1.457x | 1.405x |
| 8192 | 19.167 s (427) | 11.166 s @1024 (**733.5**) | 1.717x | 1.676x |
| 32768 | 84.544 s (388) | 44.663 s @1024 (**733.6**) | **1.893x** | 1.872x |

Chunk 1024 is the shared optimum at 8192 and 32K (the 32K sweep sits flat around
733: @1024 733.6, @1536 722, @2048 712.5). Decode is unaffected (32.3 vs 32.2
tok/s installed-cache; 28.2 vs 27.8 at 32K).

The accounting closes with ≈0 unattributed `[I61]`: box B busy fraction 77.2% →
88.1% → 94.1% across the three sizes; the remainder is the fill bubble (the
first chunk's A-stage: 0.75 / 1.33 / 2.51 s) plus starvation ≈0. The A/B split
balances to 0.3-0.6% (32/32 layers was the right cut). Activation transfers are
fully hidden; the only transfer on the critical path is cache retrieval
(345 MB / 0.26 s at 8192, 1.15 GB / 0.47 s at 32K — effective ≈2.4 GB/s), which
is what `stream_kv` then removes.

## 6. Server integration (`mlx_lm.server --prefill-2box`)

Opt-in flag, verified end-to-end `[I63]` with the full record in
`results/bench_2box/serving_verdict.json`:

- **TTFT on an 8.3K-token streaming request: 20.3 → 11.9 s (1.705x median)** —
  three alternating fresh-boot reps per arm, spread under 0.5%.
- **Multi-turn is incremental on both boxes**: turn 2 (27 new tokens) skips the
  remote hop under a 4096-token gate entirely; turn 3 (6.2K new tokens) resumes
  the runner's resident session and prefills only the increment — 10.7 s vs
  16.8 s flag-off (1.58x).
- **No-regression with the flag off**: token sequences and usage identical
  across all captures against a pristine server.
- **Failure policy is fail-fast, never silent**: the server refuses to start if
  the runner is unreachable; a mid-session runner death fails the request while
  the server survives; a restarted runner is auto-reconnected on the next
  request.
- v1 constraints, all rejected loudly rather than silently degraded: plain
  decode only (no `--mtp`, no draft model, no adapters), batching disabled
  (requests serve sequentially), and greedy outputs across *different* chunk
  schedules can flip near-ties — use `--prefill-step-size 1024` on both sides
  for schedule-matched reproducibility.

One bug class from the integration is worth naming because it recurs in MLX
threading: handing a **lazy** array slice to a sender thread crashes with
"no Stream(gpu, N) in current thread" — evaluate before enqueueing. It was found
here on the first incremental turn, and the same class had bitten a different
project of mine months earlier.

## 7. What is deliberately not claimed, and what is next

This is a **prefill** split: decode still runs on one box, so end-to-end gains
scale with how prefill-heavy the workload is (long-context ingestion, RAG,
multi-turn with big documents). Next levers, unstarted `[PA22]`: a four-stage
layer interleave to halve the fill bubble (bounded ≈+3-5%, at 3x transfers per
chunk), and a decode-side two-box (tensor-parallel over TB5) as its own campaign
with its own speculative-decoding interactions.

## Addendum (2026-08-17): the second box now pays at decode too — via speculation

The original verdict above was that TB5 tensor-parallel *decode* was not worth
it; the two-box win was prefill-only. The TP2+jaccl spike revisits that with
two things the first pass lacked: RDMA-over-Thunderbolt (jaccl backend,
`all_sum` at **21.87µs** measured on a dependency chain — the TCP ring's 459µs
made the whole idea stillborn) and the split-K kernel extended to the sharded
quantized linears (`1e22e21`), so the speculative verify widths keep their
kernel inside TP.

Measured (4-prompt canon, alternated with cooldowns, EOS-cut, same-day
controls):

| config | tok/s | vs 1-box plain |
|---|---:|---:|
| 1-box plain (control) | 35.83 | 1.00x |
| 1-box gated MTP k=4 (control) | 57.51 | 1.60x |
| TP2 plain | 48.96 | 1.37x |
| **TP2 x gated MTP** | **74.23** | **2.07x** |

Plain TP2 **fails** its 1.4x gate — and does so exactly on schedule: the
per-step 128 all-reduces at 21.87µs land within a hair of the pre-registered
break-even arithmetic. The composite **passes** its 1.8x gate with room
(74.23 vs 67.7): shrinking the per-forward time amplifies what each accepted
draft token is worth, so speculation converts a failing TP into a passing one
(+29% on top of the single-box MTP record; Korean 62.7, +22% over its
single-box cell — no regression). This is the same TPxMTP amplification
reported externally on another stack, reproduced here with paired controls.
Verification: 3 of 4 prompts token-exact over 64 tokens against the single
box, the fourth diverging at token 41 within the established fp-drift class.

Ledger `[I109]`-`[PA42]`; raw JSONs in `results/tp2_spike/`. Not integrated
into serving yet — that call, and the stage-2 lever (sharding the MTP block
and lm_head, paper arithmetic ~87 tok/s), are open.

**Stage-2 lever (sharding the MTP block + lm_head): tried, rejected
(2026-08-17).** The paper arithmetic said ~87 tok/s; a decomposition
microbench registered ~78 as the honest ceiling before the run, and the run
landed on it: composite 74.23 → **77.75 (+4.7%)**, under the +8% adoption
gate. The vocab-axis lm_head split with a full-logit all_gather kept every
contract bit-identical (verify diverges at the same token 41 as stage 1), and
plain TP2 moved +0.5% — communication was never the bottleneck. What the
rejection bought is the diagnosis: the in-loop draft cost is dominated by
chain scheduling and gate synchronization, which sharding cannot touch.
Draft-graph fusion / sync removal is the precondition for any further TP
lever, and the deep-k economics (wider verify kernel window) attacks the same
fixed cost from the other side. Ledger `[I113]`-`[PA43]`; patch preserved in
the spike directory, fork restored.

## On-demand full two-box serving (2026-08-17)

The TP2 decode result of the previous section now runs as a served stack: one
launch brings up `mlx_lm.server` across both boxes (rank 0 serves HTTP, rank 1
joins the generation loop in lockstep; requests are pickled and broadcast).
Measured over HTTP streaming, four prompts x three alternated boots, EOS-cut:

| arm | TP2 served | one box served | in-process |
|---|---:|---:|---:|
| plain, greedy | 47.84 | — | 48.96 (−2.3% tax) |
| plain, t1 | 47.88 | — | — |
| **gated MTP, greedy** | **62.90** | 53.06 (**+18.5%**) | 74.23 (−15.3% tax) |
| **gated MTP, t1** | **57.66** | 47.11 (**+22.4%**) | — |

TTFT on an 8,330-token prompt: 12.82 s with MTP, 12.57 s plain (~650-660
tok/s prefill) against 20.29 s on one box — so a single TP2 stack delivers
1.6x prefill *and* the best served decode simultaneously. The layer-pipelined
split still wins prefill outright (11.91 s, ~700 tok/s), so the choice is
workload-shaped: TP2 for interactive serving where decode dominates, the
pipeline for bulk long-prompt ingestion. Greedy output is character-identical
to the non-speculative control on all twelve cells.

**The serving tax is now the frontier.** One box paid ~0% for going through
HTTP; TP2 pays 15%. The reason is arithmetic: TP2 cuts the step to ~13 ms, so
rank 0's fixed per-token cost (SSE framing, detokenizer) stops being noise.
The next lever here is the streaming layer, not the kernel.

### The wedge rule, bought the hard way

Three separate deadlocks were chased through an RNG-divergence hypothesis and
two attempted fixes before the decisive control ran: restoring the exact code
that had worked an hour earlier **still failed**, and a reboot of the second
box made the same code pass on the first try — including the sampled arm that
had never once completed. The deadlocks were not a code defect at all; they
were the residue of an earlier collective deadlock that process teardown does
not clear. The operating rule that follows (`[RA27]`): **a box that has
suffered a distributed-collective deadlock is rebooted before the next
experiment, and failures observed in that state are not admissible evidence
about code.** Without it, hours went into fixing a defect that did not exist.

Scripts: `launch_full2box.sh` / `stop_full2box.sh` (TERM-only teardown, health
gate, smoke check). Not a resident daemon — it is brought up on demand.
Ledger `[I115]`-`[PA44]`, raw JSONs in `results/serving_full2box/`.
