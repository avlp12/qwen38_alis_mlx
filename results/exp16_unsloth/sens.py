"""imatrix 가중 양자화 오차를 텐서·비트폭별로 계산한다.

출력 채널 j, 입력 채널 i 에 대해 선형층의 출력 오차 기댓값은
    E[(Δy)²] = Σ_i E[x_i²] · Σ_j (Δw_ji)²      (입력 채널 무상관 가정)
이므로, imatrix 의 E[x_i²] 를 그대로 가중치로 쓰면 각 텐서가 출력 오차에
기여하는 양을 비트폭마다 실측할 수 있다. 그래디언트가 필요 없어
GatedDeltaNet 의 VJP 부재([CA80])에 막히지 않는다.
"""
import json, os, re, sys, time
import mlx.core as mx
import numpy as np

SRC = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/qwen38/src")
OUT = sys.argv[2] if len(sys.argv) > 2 else "sens.json"
BITS = [3, 4, 5, 6, 8]
GS = 64
D = os.path.dirname(os.path.abspath(__file__))

ex2 = {k: np.array(v, dtype=np.float64) for k, v in
       json.load(open(f"{D}/imatrix_ex2.json")).items()}

def gguf_name(mlx_name):
    """MLX 텐서 이름 → GGUF imatrix 키."""
    n = mlx_name.replace("language_model.", "").replace("model.", "")
    m = re.match(r"layers\.(\d+)\.(.+)\.weight$", n)
    if not m:
        return {"embed_tokens.weight": "token_embd.weight",
                "lm_head.weight": "output.weight",
                "mtp.fc.weight": "blk.64.nextn.eh_proj.weight"}.get(n)
    L, rest = m.group(1), m.group(2)
    tbl = {"linear_attn.in_proj_qkv": "attn_qkv", "linear_attn.in_proj_z": "attn_gate",
           "linear_attn.in_proj_a": "ssm_alpha", "linear_attn.in_proj_b": "ssm_beta",
           "linear_attn.out_proj": "ssm_out",
           "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
           "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
           "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output"}
    g = tbl.get(rest)
    return f"blk.{L}.{g}.weight" if g else None

idx = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
# 샤드 단위로 읽는다 — mx.load 가 safetensors/bf16 을 직접 다루고, 샤드당 1회 I/O 로 끝난다
shards = {}
for k, v in idx.items():
    if k.endswith(".weight"):
        shards.setdefault(v, []).append(k)

res, t0, skipped, done = {}, time.perf_counter(), 0, 0
for si, (shard, keys) in enumerate(sorted(shards.items())):
    blob = mx.load(f"{SRC}/{shard}")
    for name in sorted(keys):
        W = blob[name]
        if W.ndim != 2 or W.shape[-1] % GS != 0:
            skipped += 1; continue
        W = W.astype(mx.float32)
        gk = gguf_name(name)
        m = ex2.get(gk)
        if m is not None and len(m) == W.shape[1]:
            w = mx.array((m / m.mean()).astype(np.float32)); src_kind = "imatrix"
        else:
            w = mx.ones((W.shape[1],), dtype=mx.float32); src_kind = "uniform"
        # 분모: 같은 가중치로 잰 신호 에너지. 오차/신호 = 스케일-비의존 상대오차가 되어
        # RMSNorm 이 절대 스케일을 지우는 구조에서도 층-간 비교가 성립한다.
        sig = float(mx.sum(mx.sum(W ** 2, axis=0) * w))
        row = {"shape": list(W.shape), "numel": int(W.size), "imatrix": src_kind,
               "signal": sig, "err": {}, "bpw": {}}
        for b in BITS:
            q, sc, bi = mx.quantize(W, group_size=GS, bits=b)
            Wq = mx.dequantize(q, sc, bi, group_size=GS, bits=b)
            d2 = (W - Wq) ** 2
            row["err"][b] = float(mx.sum(mx.sum(d2, axis=0) * w))
            row["bpw"][b] = b + 32.0 / GS
        res[name] = row; done += 1
    del blob
    mx.clear_cache()
    print(f"  샤드 {si+1}/{len(shards)} · 텐서 {done} · {time.perf_counter()-t0:.0f}s", flush=True)
json.dump(res, open(f"{D}/{OUT}", "w"))
print(f"완료: {len(res)} 텐서 (조건 미달 {skipped}) → {OUT}", flush=True)
print("SENS-DONE", flush=True)
