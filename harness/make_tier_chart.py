#!/usr/bin/env python3
"""Qwen3.8-27B size-vs-fidelity tier chart — our answer to the AtomicChat chart.

Honesty rules baked in:
 · y = exact full-vocab KL to bf16 (ctx=2048 non-overlap, paired windows) — not a proxy
 · error bars = 512-token block SE pooled across slices (conservative under autocorrelation)
 · community-4bit/8bit are byte-identical to our uniform builds (verified) — plotted once
   as "uniform 4-bit/8-bit" with a shared label, so the AWQ comparison is visibly
   AWQ-vs-uniform, not us-vs-them theater
 · methodology footnote states ctx (AtomicChat used 4096; ours is 2048)
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = "../results/kl_out"  # 리포 내 상대 경로 — harness/ 에서 실행

def load(name):
    r = json.load(open(f"{D}/kl_{name}.json"))
    n = sum(s["n_tokens"] for s in r["slices"].values())
    kl = sum(s["mean_kl"] * s["n_tokens"] for s in r["slices"].values()) / n
    se = np.sqrt(sum((s["kl_block_sem"] * s["n_tokens"] / n) ** 2 for s in r["slices"].values()))
    t1 = sum(s["top1_agree_pct"] * s["n_tokens"] for s in r["slices"].values()) / n
    return kl, se, t1

#            key         label                     GB     style
BUILDS = [
    ("q8awq3",           "8-bit AWQ",             28.36, "awq_unpub"),
    ("q8v",              "8-bit uniform*",        27.90, "uniform"),
    ("q6awq3",           "6-bit AWQ",             22.00, "awq_unpub"),
    ("q6v",              "6-bit uniform",         21.53, "uniform"),
    ("q4awq3m",          "4-bit AWQ",             15.21, "awq_pub"),
    ("q4v",              "4-bit uniform*",        15.17, "uniform"),
    ("community-nvfp4",  "nvfp4 (community)",     15.01, "community"),
    ("community-mxfp4",  "mxfp4 (community)",     14.21, "community"),
]

STYLE = {
    "awq_pub":   dict(color="#0a7a3d", marker="o", s=130, zorder=5),
    "awq_unpub": dict(color="#0a7a3d", marker="o", s=90, zorder=4, facecolor="none"),
    "uniform":   dict(color="#5b6470", marker="s", s=80, zorder=3),
    "community": dict(color="#b8860b", marker="D", s=80, zorder=3),
}

fig, ax = plt.subplots(figsize=(9.2, 6.0), dpi=200)
for key, label, gb, st in BUILDS:
    kl, se, t1 = load(key)
    s = STYLE[st]
    kw = dict(color=s["color"], marker=s["marker"], s=s["s"], zorder=s["zorder"])
    if "facecolor" in s:
        kw.update(facecolors="none", edgecolors=s["color"], linewidths=1.8)
    ax.errorbar(gb, kl, yerr=se, fmt="none", ecolor=s["color"], capsize=3, zorder=2, alpha=0.7)
    ax.scatter([gb], [kl], **kw)
    dx, dy, ha = 0.35, 1.0, "left"
    if key == "q4v":
        # nvfp4 라벨(위)과 AWQ 라벨(아래) 사이 한 줄로 — 두 줄이면 좌측이 잘린다
        ax.annotate(f"{label}  KL {kl:.4f} · top-1 {t1:.1f}%",
                    (gb, kl), xytext=(gb + 0.5, kl), ha="left", va="center",
                    fontsize=8.2, color=s["color"])
        continue
    if key == "q6v":
        dx, dy, ha = -0.35, 1.0, "right"
    if key == "q8v":
        dx, dy, ha = -0.45, 0.97, "right"    # 점 왼쪽, 축 안쪽
    if key == "q8awq3":
        dx, dy, ha = 0.0, 1.55, "center"     # 오른쪽 잘림 방지 — 점 위로
    ax.annotate(f"{label}\nKL {kl:.4f} · top-1 {t1:.1f}%",
                (gb, kl), xytext=(gb + dx, kl * dy), ha=ha, va="center",
                fontsize=8.2, color=s["color"], linespacing=1.25)

# what +GB buys you (tier arrows)
kl4, _, _ = load("q4awq3m"); kl6, _, _ = load("q6v"); kl8, _, _ = load("q8v")
ax.annotate("", xy=(21.53, kl6), xytext=(15.21, kl4),
            arrowprops=dict(arrowstyle="->", color="#9aa2ad", lw=1.1, ls=":"))
ax.text(18.0, np.sqrt(kl4 * kl6) * 1.15, f"+6.3 GB → KL ÷{kl4/kl6:.0f}",
        fontsize=8.5, color="#5b6470", ha="center")
ax.annotate("", xy=(27.90, kl8), xytext=(21.53, kl6),
            arrowprops=dict(arrowstyle="->", color="#9aa2ad", lw=1.1, ls=":"))
ax.text(24.6, np.sqrt(kl6 * kl8) * 1.15, f"+6.4 GB → KL ÷{kl6/kl8:.1f}",
        fontsize=8.5, color="#5b6470", ha="center")

ax.set_yscale("log")
ax.set_xlabel("Model size on disk (GB)", fontsize=10.5)
ax.set_ylabel("KL divergence to bf16 (exact, full vocab; log scale)", fontsize=10.5)
ax.set_title("Qwen3.8-27B on Apple Silicon — what each GB buys you",
             fontsize=13, pad=14, weight="bold")
ax.grid(True, which="both", alpha=0.25, lw=0.5)
ax.set_xlim(13.0, 31.5)
yb = ax.get_ylim()
ax.set_ylim(yb[0] * 0.75, yb[1])

from matplotlib.lines import Line2D
legend = [
    Line2D([], [], color="#0a7a3d", marker="o", ls="", ms=10, label="AWQ, vision+MTP preserved (published 4-bit)"),
    Line2D([], [], color="#0a7a3d", marker="o", ls="", ms=8, markerfacecolor="none", label="AWQ (measured, unpublished)"),
    Line2D([], [], color="#5b6470", marker="s", ls="", ms=8, label="uniform gs64 (*byte-identical to mlx-community)"),
    Line2D([], [], color="#b8860b", marker="D", ls="", ms=8, label="mlx-community fp4 variants"),
]
ax.legend(handles=legend, fontsize=8.2, loc="lower left", framealpha=0.9)

fig.text(0.99, 0.005,
         "KL(bf16‖build) per token, en+ko+code corpus (100k tokens, paired windows, ctx 2048); "
         "error bars = 512-token block SE. bf16 = 51.8 GB.",
         fontsize=7, color="#8a8f98", ha="right")
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(f"{D}/tier_chart.png", bbox_inches="tight")
print("saved", f"{D}/tier_chart.png")
