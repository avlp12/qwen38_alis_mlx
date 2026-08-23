"""DSv4-Flash TP2 스모크: omlx 오버레이 + model.shard(group) + 플레인 greedy.
Pro 레시피(run_tp4_pro0813_dspark.py)의 각색 — DSpark 단언을 선택화."""
import argparse, os, sys, time

os.environ.setdefault("MLX_METAL_FAST_SYNCH", "0")
import mlx.core as mx

sys.path.insert(0, "/Users/Shared/tp2")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/dsv4flash/mlx4bit"))
    ap.add_argument("--prompt", default="17 * 23 =")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--require-world", type=int, default=2)
    args = ap.parse_args()

    group = mx.distributed.init()
    rank, world = group.rank(), group.size()
    print(f"[rank {rank}] world={world}", flush=True)
    assert world == args.require_world, f"world {world} != {args.require_world}"

    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch
    apply_deepseek_v4_patch()
    from mlx_lm import load
    from dspark_tp4_common import shard_mtp

    t0 = time.monotonic()
    model, tok = load(args.model, lazy=True)
    assert hasattr(model, "shard"), "omlx base patch 미적용"
    model.shard(group)
    n_mtp = 0
    try:
        n_mtp = shard_mtp(model, group)
    except Exception as e:
        if rank == 0:
            print(f"[tp2] mtp 샤딩 생략: {e}", flush=True)
    # 대형모델 하네스 규칙: wired limit 필수 + 층별 단계 실체화(메모리 스파이크 방지)
    info = mx.metal.device_info()
    mx.set_wired_limit(info["max_recommended_working_set_size"])
    inner = getattr(model, "model", model)
    heads = [m for m in (getattr(inner, n, None) for n in ("embed_tokens", "norm", "hc_head")) if m]
    if getattr(model, "lm_head", None) is not None:
        heads.append(model.lm_head)
    if heads:
        mx.eval(*[m.parameters() for m in heads])
    for i, layer in enumerate(model.model.layers):
        mx.eval(layer.parameters())
        mx.synchronize()
        if rank == 0 and (i + 1) % 8 == 0:
            print(f"[tp2-load] layer {i+1}/{len(model.model.layers)}", flush=True)
    mx.eval(model.parameters())
    mx.synchronize()
    heads = int(model.model.layers[0].attn.n_heads)
    if rank == 0:
        print(f"[tp2] load+shard {time.monotonic()-t0:.1f}s · layers "
              f"{len(model.model.layers)} · heads/rank {heads} · mtp {n_mtp}", flush=True)

    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    mx.random.seed(7)  # 전 랭크 동일 시드 (JACCL 요건)
    msgs = [{"role": "user", "content": args.prompt}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True)
    out = []
    for r in stream_generate(model, tok, text, max_tokens=args.max_tokens,
                             sampler=make_sampler(0.0)):
        out.append(r.text)
    if rank == 0:
        print(f"[tp2] prompt_tps {r.prompt_tps:.1f} · gen_tps {r.generation_tps:.2f}")
        print("[tp2] OUT:", "".join(out)[:200].replace("\n", " "))
        print("[tp2-flash-pass]", flush=True)

if __name__ == "__main__":
    main()
