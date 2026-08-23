"""jaccl all_sum 의존-사슬 지연 마이크로벤치 (qwen38 캠페인 방법 재현)."""
import os, time
import mlx.core as mx
g = mx.distributed.init()
r = g.rank()
x = mx.ones((1, 4096), dtype=mx.bfloat16)  # 디코드 은닉 크기
mx.eval(x)
# 워밍업
for _ in range(20):
    x = mx.distributed.all_sum(x, group=g) * 0.5
mx.eval(x); mx.synchronize()
N = 300
t0 = time.perf_counter()
for _ in range(N):
    x = mx.distributed.all_sum(x, group=g) * 0.5  # 의존 사슬
mx.eval(x); mx.synchronize()
dt = (time.perf_counter() - t0) / N * 1e6
if r == 0:
    print(f"FAST_SYNCH={os.environ.get('MLX_METAL_FAST_SYNCH','미설정')} · all_sum 사슬 {dt:.1f} µs/op")
print("LAT-DONE" if r == 0 else "", flush=True)
