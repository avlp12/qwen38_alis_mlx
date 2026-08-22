"""FreeToken 규칙의 두 입력을 uran 에서 직접 잰다.

  q* = m * B_P / B_H
     B_P: 고정(pinned) 호스트→디바이스 PCIe 대역폭 — 전문가를 VRAM 으로 채우는 경로
     B_H: 호스트측 전문가 처리 대역폭 — CPU 가 호스트 메모리의 전문가를 그대로 도는 경로

논문의 전제는 B_P << B_H 다. 두 값이 비슷하면 나눌 이유가 없고, 규칙도 무의미해진다.
"""
import ctypes, time, numpy as np
from cuda.bindings import runtime as cudart

def ck(r):
    err = r[0]
    if int(err) != 0:
        raise RuntimeError(f"CUDA error {err}")
    return r[1:] if len(r) > 2 else (r[1] if len(r) == 2 else None)

N = 512 << 20          # 512 MiB
REP = 8
free_, total_ = ck(cudart.cudaMemGetInfo())
print(f"VRAM total {total_/2**30:.1f} GiB · free {free_/2**30:.1f} GiB", flush=True)

h = ck(cudart.cudaHostAlloc(N, cudart.cudaHostAllocDefault))
d = ck(cudart.cudaMalloc(N))
ck(cudart.cudaMemcpy(d, h, N, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
ck(cudart.cudaDeviceSynchronize())
best = 1e18
for _ in range(REP):
    t0 = time.perf_counter()
    ck(cudart.cudaMemcpy(d, h, N, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
    ck(cudart.cudaDeviceSynchronize())
    best = min(best, time.perf_counter() - t0)
BP = N / best / 1e9
print(f"B_P  pinned H2D PCIe : {BP:8.2f} GB/s", flush=True)

best = 1e18
for _ in range(REP):
    t0 = time.perf_counter()
    ck(cudart.cudaMemcpy(h, d, N, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
    ck(cudart.cudaDeviceSynchronize())
    best = min(best, time.perf_counter() - t0)
print(f"     pinned D2H PCIe : {N/best/1e9:8.2f} GB/s", flush=True)

# B_H: 호스트 메모리 읽기 대역폭 (전문가 가중치를 CPU 가 훑는 속도의 상한)
a = np.ones(N // 4, dtype=np.float32)
b = np.empty_like(a)
np.copyto(b, a)
best = 1e18
for _ in range(REP):
    t0 = time.perf_counter(); np.copyto(b, a); best = min(best, time.perf_counter() - t0)
BH_copy = 2 * a.nbytes / best / 1e9        # 읽기+쓰기
best = 1e18
for _ in range(REP):
    t0 = time.perf_counter(); s = float(a.sum(dtype=np.float32)); best = min(best, time.perf_counter() - t0)
BH_read = a.nbytes / best / 1e9
print(f"B_H  host copy (R+W): {BH_copy:8.2f} GB/s", flush=True)
print(f"B_H  host read      : {BH_read:8.2f} GB/s", flush=True)
print(f"\n비 B_P/B_H(read) = {BP/BH_read:.3f}  →  m개 미스 중 채울 몫 q* = {BP/BH_read:.3f}·m", flush=True)
ck(cudart.cudaFree(d)); ck(cudart.cudaFreeHost(h))
print("BW-DONE", flush=True)
