"""호스트 메모리 대역폭을 다중 스레드로 제대로 잰다(STREAM 식).
numpy 의 copyto/sum 은 GIL 을 놓으므로 스레드가 실제로 병렬로 돈다."""
import time, numpy as np, threading, os
N = 1 << 30           # 1 GiB
a = np.ones(N // 4, dtype=np.float32)
b = np.empty_like(a)

def bw(nthreads, rep=5):
    chunks = np.array_split(np.arange(len(a)), nthreads)
    idx = [(c[0], c[-1] + 1) for c in chunks]
    best = 1e18
    for _ in range(rep):
        ths = [threading.Thread(target=lambda s, e: np.copyto(b[s:e], a[s:e]), args=(s, e))
               for s, e in idx]
        t0 = time.perf_counter()
        [t.start() for t in ths]; [t.join() for t in ths]
        best = min(best, time.perf_counter() - t0)
    return 2 * a.nbytes / best / 1e9      # 읽기+쓰기

print(f"논리 코어 {os.cpu_count()}", flush=True)
peak = 0
for n in (1, 2, 4, 8, 12, 16, 24):
    v = bw(n); peak = max(peak, v)
    print(f"  스레드 {n:>3}: {v:7.2f} GB/s (R+W)", flush=True)
print(f"\nB_H 최대 관측 {peak:.2f} GB/s (R+W) · 유효 읽기 대역폭 ~{peak/2:.2f} GB/s", flush=True)
BP = 49.70
print(f"B_P = {BP:.2f} GB/s → B_P/B_H(R+W) = {BP/peak:.3f} · B_P/유효읽기 = {BP/(peak/2):.3f}", flush=True)
print("HOSTBW-DONE", flush=True)
