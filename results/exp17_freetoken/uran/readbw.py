"""순수 읽기 대역폭 — FreeToken 의 두 경로가 **같은 호스트 메모리를 읽는다**는 점이 핵심.
전문가를 PCIe 로 보내든 CPU 로 돌리든 먼저 호스트에서 읽어야 하므로, 읽기 대역폭이
두 경로의 공통 상한이다. 논문의 T_cpu 분모가 (B_H - B_P) 인 이유도 그것이다."""
import time, numpy as np, threading, os
N = 1 << 30
a = np.ones(N // 4, dtype=np.float32)
out = [0.0] * 64

def read_bw(nthreads, rep=5):
    bnds = np.linspace(0, len(a), nthreads + 1).astype(int)
    def work(i, s, e):
        out[i] = float(np.add.reduce(a[s:e]))     # 읽기 전용, GIL 해제
    best = 1e18
    for _ in range(rep):
        ths = [threading.Thread(target=work, args=(i, bnds[i], bnds[i+1])) for i in range(nthreads)]
        t0 = time.perf_counter(); [t.start() for t in ths]; [t.join() for t in ths]
        best = min(best, time.perf_counter() - t0)
    return a.nbytes / best / 1e9

peak = 0
for n in (1, 4, 8, 16, 24):
    v = read_bw(n); peak = max(peak, v)
    print(f"  읽기 스레드 {n:>3}: {v:7.2f} GB/s", flush=True)
BP = 49.70
print(f"\nB_H(순수 읽기) 최대 {peak:.2f} GB/s · B_P = {BP:.2f} GB/s", flush=True)
print(f"B_H - B_P = {peak - BP:+.2f} GB/s  → CPU 경로에 남는 대역폭 "
      f"{'있음' if peak > BP else '없음'}", flush=True)
print("READBW-DONE", flush=True)
