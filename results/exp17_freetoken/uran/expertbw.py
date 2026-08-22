"""B_H 를 '전문가 처리' 연산으로 직접 확인한다.

디코드(배치 1)에서 전문가 실행은 y = x @ W 인 **행렬-벡터** 곱이라 순수 메모리-바운드다.
따라서 CPU 의 전문가 처리 대역폭은 W 를 훑는 속도와 같아야 한다 — 앞서 잰 순수 읽기
62.73 GB/s 가 맞는 값인지, 아니면 실제 GEMV 가 그보다 훨씬 느린지가 q* 를 좌우한다.
(느리면 CPU 경로는 무가치해지고 q* → 1 로 퇴화한다.)
"""
import time, numpy as np, threading, os
H, I, NEXP = 7168, 2048, 16          # 대표적 MoE 전문가 형상, 16개분
W = [np.ascontiguousarray(np.random.randn(I, H).astype(np.float16)) for _ in range(NEXP)]
x = np.random.randn(H).astype(np.float16)
bytes_total = sum(w.nbytes for w in W)
print(f"전문가 {NEXP}개 · 개당 {W[0].nbytes/2**20:.1f} MiB · 합계 {bytes_total/2**20:.0f} MiB", flush=True)

def run(nthreads, rep=5):
    idx = np.array_split(np.arange(NEXP), nthreads)
    res = [None]*nthreads
    def work(i, ids):
        acc = None
        for j in ids:
            acc = W[j].astype(np.float32) @ x.astype(np.float32) if acc is None else acc
        res[i] = acc
    best = 1e18
    for _ in range(rep):
        ths=[threading.Thread(target=work, args=(i, idx[i])) for i in range(nthreads)]
        t0=time.perf_counter(); [t.start() for t in ths]; [t.join() for t in ths]
        best=min(best, time.perf_counter()-t0)
    return bytes_total/best/1e9

# fp16 캐스팅 비용을 빼기 위해 fp32 사본으로도 잰다
W32 = [w.astype(np.float32) for w in W]; x32 = x.astype(np.float32)
b32 = sum(w.nbytes for w in W32)
def run32(nthreads, rep=5):
    idx = np.array_split(np.arange(NEXP), nthreads); res=[None]*nthreads
    def work(i, ids):
        for j in ids: res[i] = W32[j] @ x32
    best=1e18
    for _ in range(rep):
        ths=[threading.Thread(target=work, args=(i, idx[i])) for i in range(nthreads)]
        t0=time.perf_counter(); [t.start() for t in ths]; [t.join() for t in ths]
        best=min(best, time.perf_counter()-t0)
    return b32/best/1e9

print("fp32 GEMV (전문가 처리 대역폭):", flush=True)
peak=0
for n in (1,4,8,16,24):
    v=run32(n); peak=max(peak,v); print(f"  스레드 {n:>3}: {v:7.2f} GB/s", flush=True)
BP, BH_read = 49.70, 62.73
print(f"\nB_H(GEMV 실측) {peak:.2f} GB/s · B_H(순수 읽기) {BH_read:.2f} · B_P {BP:.2f}", flush=True)
eff = min(peak, BH_read)
print(f"q* = m·B_P/B_H = {BP/eff:.3f}·m  → CPU 몫 {max(0,1-BP/eff):.1%}", flush=True)
print("EXPERT-DONE", flush=True)
