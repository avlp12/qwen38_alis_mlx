# exp12 — the wide-M kernel window, and what it was worth

Raw results behind kernels.md §5 and ledger nodes [I129]–[I133], [RA33]–[RA35],
[CA18], [PA46].

| file | what it is |
|---|---|
| `off_r{1,2}.json` | gated MTP k=4/6/8, four prompts, wide kernel **off** (epsilon, quiet) |
| `on_r{1,2}.json` | the same arms with `MLXLM_FAST_QMM_WIDE=1` |
| `ab_deepk.log` | the alternated run log for both rotations |
| `width_hist.json` | calls by M for k=4/6/8 (`MLXLM_QMM_HIST=1`) — the residency measurement |
| `g_wide_on.json` | DSpark `max_width` 8 / 12 / 16 / unbounded, wide kernel on (gesicht) |

Read the k=4 and k=6 arms as **null controls**: those verify at widths 5 and 7,
which the wide kernel cannot structurally reach, so their movement is the
harness's own noise (±5-8% per cell). That is why k=8's +4.1% is reported as
unclaimable rather than as a gain.

`g_wide_on.json` carries `mean_width` and `max_width_obs` per cell — the
unbounded arm drifts to a mean width of 30.9, well past the kernel's reach.
