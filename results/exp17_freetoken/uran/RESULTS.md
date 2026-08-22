# uran (Windows, RTX 5090, TB4 to gesicht) — FreeToken 전제 실측

측정일 2026-08-22. 도구: `pcie_bw.py`(cuda-python), `hostbw.py`, `readbw.py`, `tcpbw.py`.

## 하드웨어
- CPU Intel Core Ultra 9 285K (24 논리 코어) · RAM **255 GB** · Windows 11 Pro
- GPU RTX 5090, VRAM 31.8 GiB (여유 29.7) · 드라이버 610.47 · CUDA 툴체인 13.2
- gesicht(192.168.7.2) 와 TB4 데이터 플레인 직결(192.168.7.1, bridge0)

## FreeToken 규칙의 두 입력
| 양 | 실측 | 논문 참조값 |
|---|---|---|
| **B_P** 고정 H2D PCIe | **49.70 GB/s** | 약 52 GB/s |
| D2H PCIe | 38.25 GB/s | – |
| **B_H** 호스트 순수 읽기(24스레드) | **62.73 GB/s** | 50–178 GB/s |
| 호스트 copy(R+W, 24스레드) | 66.14 GB/s | – |

스레드 확장: 읽기 1→9.28, 4→28.34, 8→37.59, 16→44.24, **24→62.73 GB/s**.
단일 스레드로 재면 8-9 GB/s 라 규칙이 무의미해 보인다 — **다중 스레드로 재야 한다.**

## 판정
`B_H − B_P = +13.03 GB/s` → CPU 경로에 잔여 대역폭이 있다.
`q* = m·B_P/B_H = 0.792·m` — 미스 전문가의 **79% 는 PCIe 로 채우고 21% 는 CPU 에서** 실행.
PCIe 만 쓰는 경우 대비 전문가 인출 단계 **1.26배**(62.73/49.70).

**전제는 성립한다.** 다만 uran 의 호스트 대역폭은 논문 범위의 **하단**이라 PCIe-채움 쪽으로
치우친다 — 178 GB/s 짜리 서버라면 q\* 는 0.28 로 CPU 몫이 훨씬 커진다.

## B_H 를 '전문가 처리' 연산으로 재확인
순수 읽기는 상한일 뿐이므로, 대표 MoE 전문가 형상(W[2048,7168] fp32, 16개=448 MiB)의
**행렬-벡터 곱**으로 다시 쟀다. 배치 1 디코드의 전문가 실행은 순수 메모리-바운드라
훑는 속도와 같아야 한다.

| 스레드 | 1 | 4 | 8 | 16 | 24 |
|---|---|---|---|---|---|
| GEMV GB/s | 61.49 | 67.99 | **68.76** | 68.57 | 67.18 |

**68.76 GB/s** — 순수 읽기 62.73 과 같은 급이다. 즉 앞선 측정이 맞았고,
`q* = 0.792·m` (CPU 몫 **20.8%**) 는 대리값이 아니라 해당 연산으로 확인된 값이다.

## ⚠ 실행 차단 — FreeToken CLI 는 리눅스 전용
`uv pip install "freetoken[accel]"` 실패. 원인 두 겹:
- `triton` 에 Windows 휠이 없다(manylinux 만).
- `freetoken` **0.1.2 자체가 manylinux_2_27_x86_64 휠만** 제공한다.
README 의 Windows 지원은 데스크톱 앱(flashml.ai) 쪽이고 CLI 가 아니다.

uran 에 **WSL 이 설치돼 있지 않다**("Linux용 Windows 하위 시스템이 설치되어 있지
않습니다"). `wsl --install` 은 관리자 권한 + 재부팅이 걸린 **시스템 변경**이라
임의로 하지 않았다. 선택지:
1. 사용자가 `wsl --install` (관리자, 재부팅) → 그 뒤 CLI 설치·벤치 진행
2. flashml.ai 데스크톱 앱(Windows 네이티브, GUI — ssh 헤드리스 벤치에 부적합)
3. FreeToken 대신 **Windows 네이티브 CUDA 베이스라인**(llama.cpp CUDA 빌드는 MoE
   CPU 오프로드 `--n-cpu-moe` 지원)으로 uran 의 실제 서빙 수치를 먼저 확보

## TB4 링크 (gesicht ↔ uran)
단일 스트림 파이썬 TCP 로 **0.63 GB/s** (8 GiB / 13.72 s). 이는 **하한**이며 링크 능력이
아니다(GIL·단일 스트림·4 MiB 청크). 그럼에도 uran 자신의 PCIe(49.7)보다 두 자릿수 배
좁으므로, 두 기계를 층-파이프로 묶는 것은 매력이 없다. 프레임워크도 다르다(MLX 대 CUDA).
인바운드는 Windows 방화벽에 막혀 있어 **uran 이 나가는 방향으로만** 측정했다 —
방화벽 규칙은 건드리지 않았다.

## 아직 안 한 것
FreeToken **시스템 자체는 돌리지 않았다.** 여기서 확인한 것은 그 규칙이 요구하는 두
대역폭이 이 기계에서 어떤 값을 갖고, 분할이 퇴화하지 않는다는 것까지다.

## llama.cpp 기준선 (Windows 네이티브 CUDA, WSL 불필요)

빌드: 공식 `ggml-org/llama.cpp` 릴리스 **b10569**,
`llama-b10569-bin-win-cuda-13.3-x64.zip`(140.1 MiB) + `cudart-...-13.3-x64.zip`(372.9 MiB).
CUDA 백엔드 정상 로드, RTX 5090 인식(VRAM 30,991 MiB 여유). 회선 63.7 MB/s.

### 1) Qwen3.6-35B-A3B MXFP4 — 전량 VRAM (스택 검증)
`llama-bench -p 512 -n 128 -ngl 99 -r 3`, 모델 20.21 GiB / 34.66 B

| 항목 | 값 |
|---|---|
| 프리필 pp512 | **9277.90 ± 1980.82 tok/s** |
| 디코드 tg128 | **249.60 ± 2.48 tok/s** |

**주의 — 이건 FreeToken 주장과 직접 대조가 아니다.** 이 구성은 모델이 32GB VRAM 에
통째로 들어가므로 CPU 오프로드 경로를 전혀 쓰지 않는다. 논문의 기여가 걸린 지점이 아니다.
RTX 5090 에 대한 그들의 공개 수치는 **DeepSeek-V4-Flash 284B 22–25 tok/s** 이며,
그 모델은 VRAM 에 들어가지 않는다 — 그것이 대조 대상이다.

## iGPU 를 호스트 경로에 추가하면? (사용자 제안)

uran 은 맥보다 층이 많다 — RTX 5090(전용 VRAM) + **Intel Graphics iGPU(시스템 RAM 공유,
OpenCL 상 64 CU / 전역 136.4 GiB)** + **Intel AI Boost NPU** + CPU 24 코어.
FreeToken 은 호스트 쪽에서 **CPU 만** 쓴다.

핵심 물리: 배치 1 디코드의 전문가 실행은 **순수 메모리-바운드**다. 계산 유닛을 더해도
**총 읽기 대역폭이 늘지 않으면 이득이 0** 이다. 그래서 직접 쟀다.

### 메모리 구성
4×64 GB DDR5, 정격 6400 이나 **5600 MT/s 로 구성** → 듀얼채널 이론 최대 **89.6 GB/s**.

### 고정-창(6s) 측정
| 구성 | CPU | iGPU | 합계 |
|---|---|---|---|
| CPU 단독 | 63.03 | – | 63.03 |
| iGPU 단독 | – | 52.60 | 52.60 |
| **동시** | 39.56 | 32.79 | **72.35** |

- 이론 최대의 **81%** — 상한 아래, 물리적으로 정합.
- 단순합 115.63 대비 실제 72.35 → **경합 손실 37.4%**.
- **CPU 단독 대비 +14.8%** (+9.32 GB/s).

### 판정
**iGPU 는 대역폭을 더한다. 단 +14.8% 다.** CPU 코어만으로는 DRAM 을 70% 밖에 못 채우고
있었고 iGPU 가 남은 몫을 먹는다. FreeToken 식에 넣으면 B_H 63.03→72.35 이므로
`q*` 가 0.789→0.687 로 내려가고(호스트 몫 21.1%→31.3%), 전문가 인출 단계가 **1.15배**
빨라진다.

### 계측기 함정 두 번 (둘 다 결론을 뒤집을 뻔했다)
1. **비합병 커널** — 처음엔 워크아이템마다 8192 float 연속 구간을 훑게 짜서 iGPU 가
   **6.02 GB/s** 로 나왔고, "iGPU 는 같은 파이프를 나눠 쓸 뿐 무익"이라는 정반대 결론이
   나올 뻔했다. 그리드-스트라이드(인접 워크아이템 = 인접 주소)로 고치자 52.53 GB/s.
2. **비겹침 최소-시간** — 두 워커가 각자 최소-시간 반복을 골라, 실제로 겹치지 않은 구간의
   값을 합산했다. 합계 113.44 GB/s 로 **이론 최대 89.6 을 27% 초과** — 불가능한 값이
   경보가 됐다. 고정 벽시계 창에서 각자 옮긴 바이트를 세는 방식으로 교정.

### 아직 안 한 것
이것은 **대역폭 측정이지 서빙 측정이 아니다.** llama.cpp 나 FreeToken 이 CPU 와 iGPU 에
전문가 GEMV 를 **동시에** 쪼개 줄 수 있는지는 별개의 공학 문제다(llama.cpp 은 백엔드별
빌드라 CUDA+SYCL/Vulkan 동시 사용에 RPC 같은 우회가 필요하다). NPU 는 미측정.

## DeepSeek-V4-Flash 284B 오프로드 스윕 (llama.cpp 기준선)

모델 `unsloth/DeepSeek-V4-Flash-GGUF` **UD-Q4_K_XL**, 5샤드 144.44 GiB
(HF 원본과 바이트 일치 확인). llama.cpp 이 헤더를 **"deepseek4 ?B MXFP4 MoE / 284.33 B"**
로 읽는다 — 전문가가 MXFP4 라 FreeToken 이 돌린 "극단적 양자화 없는" 구성과 급이 맞는다.
`llama-bench -p 512 -n 128 -ngl 99 -fa on -r 2`, `-ncmoe N` = 층 N 개의 전문가를 CPU 로.

| ncmoe | pp512 | tg128 |
|---|---|---|
| 61 (전량 CPU) | 40.15 ± 6.49 | 13.73 ± 0.18 |
| 55 | 37.46 ± 2.34 | 13.70 ± 0.18 |
| 50 | 46.07 ± 7.48 | 13.75 ± 0.03 |
| 45 | 45.84 ± 7.53 | 13.72 ± 0.07 |
| **40** | 42.49 ± 6.32 | **14.59 ± 0.12** |
| 35 | 16.67 ± 0.45 | 11.14 ± 0.05 |

**최고 디코드 14.59 tok/s (ncmoe=40).** 61→40 구간이 거의 평평하다 — 전문가 대부분이
여전히 호스트에 있어 병목이 그대로다. 35 에서 무너지는 것은 VRAM 압박이다.

### FreeToken 공개치와 대조
| | 디코드 |
|---|---|
| llama.cpp 기준선 (uran, 실측) | **14.59 tok/s** |
| FreeToken (RTX 5090, 저자 공개) | 22–25 tok/s |
| 비 | **1.51–1.71×** |

그들이 주장한 "1.8–2.3× vs 최강 베이스라인"보다는 낮지만 같은 자릿수다.

### 격차의 정체는 q* 분할이 아니다
호스트 읽기 63 GB/s 에서 14.59 tok/s 면 **토큰당 약 4.3 GB 를 호스트에서 읽는다**는 뜻이다.
CPU+iGPU(72.35 GB/s)로 올려도 16.8 tok/s, **+15%** 에 그친다 — 22–25 에 못 미친다.
즉 그들의 우위는 **호스트를 더 빨리 읽는 데**서 오지 않는다.

남는 설명은 **동적 전문가 상주**다. llama.cpp 의 `-ncmoe` 는 "층 0..N 의 전문가는 항상
CPU" 라는 **정적 배치**이고, FreeToken 은 **라우터를 따라가는 LRU 캐시**라 연속 토큰이
겹치는 전문가를 부르는 성질(시간적 지역성)을 먹는다. 캐시가 맞으면 호스트 읽기가
**아예 없다**. 대역폭을 넓히는 것과 읽을 일을 없애는 것의 차이다.

→ 우리가 이식할 값어치가 있는 것도 q* 가 아니라 **이쪽**이다(미착수).

## NPU 는 얼마나 기여하나

OpenVINO 2026.3.0 이 uran 의 네 장치를 모두 노출한다:
`CPU` / `GPU.0`(Intel Graphics, iGPU) / `GPU.1`(RTX 5090) / **`NPU`(Intel AI Boost)**.

전문가 GEMV 형상(W[7168,8192] fp16 = 112 MiB, 온칩 SRAM 보다 훨씬 큼)을 반복 호출해
**가중치 스트리밍 대역폭**으로 잰다. 고정 창 6s, 완료 추론 수 × W 바이트.

| 구성 | CPU | iGPU | NPU | 합계 |
|---|---|---|---|---|
| CPU 단독 | 36.47 | – | – | 36.47 |
| iGPU 단독 | – | 53.42 | – | 53.42 |
| **NPU 단독** | – | – | **48.00** | **48.00** |
| CPU+iGPU | 20.99 | 33.96 | – | 54.95 |
| **CPU+iGPU+NPU** | 19.03 | 21.16 | 18.52 | **58.71** |

- **NPU 단독은 iGPU 와 맞먹는다**(48.00 대 53.42) — 저전력 유닛치고 훨씬 세다.
- 그러나 **더했을 때 기여는 +3.76 GB/s (+6.9%)** 다.
- 단독 단순합 137.89 대비 실제 3자 58.71 → **경합 손실 57.4%**. 세 유닛이 **하나의
  메모리 컨트롤러**를 나눠 쓴다.

### 상한이 답을 미리 정해둔다
DDR5-5600 듀얼채널 이론 최대 **89.6 GB/s**. 원시 스트리밍 기준 CPU 단독 63.03(70%),
CPU+iGPU 72.35(81%). **남은 여유는 17.25 GB/s** 뿐이므로, 무엇을 더 붙이든
CPU 단독 대비 **최대 +42%**, CPU+iGPU 이후로는 **+24%가 절대 상한**이다.

### 디코드 환산 (토큰당 호스트 읽기 4.3 GB 기준)
| 호스트 경로 | 예상 디코드 |
|---|---|
| CPU 만 (현 llama.cpp) | 14.59 tok/s (실측) |
| + iGPU | 약 16.8 (+15%) |
| + NPU | 약 17.9 (+23%) |
| FreeToken 공개치 | **22–25** |

**가속기를 전부 동원해도 닿지 않는다.** [RA74] 재확인 — 그들의 우위는 호스트를 빨리
읽는 데 있지 않고 **LRU 상주로 읽을 일을 없애는 데** 있다.

### 공학적 판단
NPU 를 실제로 쓰려면 OpenVINO 정적 그래프에 **토큰마다 바뀌는 라우팅**을 얹어야 하는데,
그 대가로 얻는 게 +6.9% 면 수지가 맞지 않는다. iGPU(+15%)가 그나마 값어치가 있고,
진짜 레버는 동적 상주다.

※ 절대값 주의: 원시 스트리밍(OpenCL)은 CPU+iGPU 72.35, OpenVINO GEMV 는 58.71 로
다르다 — GEMV 쪽은 호출당 오버헤드가 섞인다. **결론의 모양(유닛 추가는 체감이 급감)은
두 하네스가 일치**한다.

## PCIe 까지 넣은 4자 동시 측정 — "가속기 전부 + FreeToken" 가설

미스 전문가는 어느 경로로 가든 DRAM 을 한 번 통과한다. PCIe DMA 도 메모리 컨트롤러
손님이므로, '전송'과 '호스트 계산'은 **같은 파이프를 나눠 쓴다**. 고정 창 6s:

| 구성 | CPU | iGPU | NPU | PCIe | 합계 |
|---|---|---|---|---|---|
| PCIe 단독 | – | – | – | 49.51 | 49.51 |
| **CPU+PCIe** (FreeToken 현재) | 25.12 | – | – | 28.57 | **53.68** |
| CPU+iGPU+PCIe | 14.43 | 31.26 | – | 15.55 | 61.25 |
| **전부 동원** | 13.26 | 19.28 | 16.45 | 14.03 | **63.02** |

**CPU+PCIe 대비 +9.33 GB/s = +17.4%** · 천장(89.6) 대비 70.3%.

### 결정적 부수 확인
**PCIe 단독 49.51 이 CPU 가 끼어들자 28.57 로 반토막** 난다. 전송과 호스트 계산이 같은
컨트롤러를 다툰다는 직접 증거다. 이는 FreeToken 식의 가정에도 시사점이 있다 —
그들의 `T_cpu` 분모 `(B_H − B_P)` 는 **PCIe 가 우선권을 갖고 CPU 가 잔여를 쓴다**는
비대칭 가정인데, 우리 측정은 훨씬 **대칭적**이다(PCIe 49.51→28.57, CPU 36→25).

### 디코드 환산
FreeToken 22–25 tok/s 가 그 지점에서 대역폭-바운드라면 53.68→63.02 로 **약 26–29 tok/s**.
**두 배가 되지 않는다 — 유닛이 모자란 게 아니라 파이프가 하나이기 때문이다.**

### 성립 조건 셋
1. FreeToken 이 그 지점에서 **정말 대역폭-바운드**여야 한다. LRU 가 미스를 충분히 줄여
   놓았다면 병목이 다른 데 있고 대역폭을 더 줘도 안 먹는다.
2. **FreeToken 에 iGPU/NPU 백엔드가 없다.** 호스트 경로는 CPU 전용이라 OpenVINO 백엔드를
   새로 다는 작업이 필요하다.
3. 전부 동원해도 천장의 70.3% — 남은 30% 는 호출당 오버헤드로 보이며, 잘 짜면 여지가 있다.

## FreeToken 실제 구동 (2026-08-23)

### 설치 경로 (재현 가능)
WSL2 + Ubuntu 26.04 → uv → **CUDA 13.3** → `ninja-build`, `build-essential` →
`uv pip install "freetoken[accel]"` (0.1.2). WSL 메모리는 `.wslconfig` 로 200GB 상향
(기본 125GB 로는 148.7 GiB 체크포인트가 안 들어간다).

**함정: CUDA 13.1 로는 JIT 가 실패한다.** Ubuntu 26.04 의 **glibc 2.43** 이 C23 `rsqrt`/
`rsqrtf` 를 선언하는데 CUDA 13.1 헤더의 예외 명세와 충돌한다
(`exception specification is incompatible with that of previous function "rsqrt"`).
`-ccbin` 으로 gcc-13 을 물려도 tvm-ffi 가 무시한다. **CUDA 13.3 에서 해소.**
apt 저장소(`wsl-ubuntu`)에 13.3 까지 있다.

### `ft bench bw` — 그들 자신의 보정값
| 양 | FreeToken | 내 실측 |
|---|---|---|
| CPU STREAM 읽기 | **73.16 GB/s** | 68.76 (GEMV) / 62.73 (순수) |
| PCIe H2D | **50.78** | 49.70 |
| PCIe D2H | **38.30** | 38.25 |
PCIe 는 2% 안에서 일치. CPU 는 그들이 7% 높다 — 포맷별 전용 커널을 쓴다
(`cpu_moe_isa: avx2`, nvfp4 는 `avx2+vnni(nvfp4-w4a8)`).

### 그들은 식이 아니라 **겹침을 직접 잰다**
| 포맷 | CPU 단독 | PCIe 단독 | 겹침 실측 | PCIe 몫 |
|---|---|---|---|---|
| bf16 | 65.16 | 50.00 | CPU 36.18 + PCIe 32.08 | **47.0%** |
| nvfp4 | 53.35 | 49.04 | CPU 18.07 + PCIe 38.18 | 67.9% |
| fp8_block | — | 48.17 | (CPU 경로 없음) | 100% |
| mxfp4 | 34.27 | 49.31 | CPU 14.47 + PCIe 40.63 | 73.7% |
| ds_fp4 | 55.84 | 49.44 | CPU 32.39 + PCIe 33.12 | 50.6% |

bf16 에서 단순합 115.2 가 겹치면 68.3 으로 떨어지는 것을 그대로 측정해 배분에 쓴다 —
내가 [I198]/[I201] 에서 잰 경합 손실과 같은 현상이며, **그들은 이미 실측으로 처리한다.**
포맷마다 PCIe 몫이 47~74% 로 갈리는 것은 닫힌 식 하나로는 나오지 않는 값이다.
**`fp8_block` 은 CPU MoE 경로가 아예 없어 하이브리드가 불가**하고 전량 PCIe 로 간다 —
이는 곧 잴 DeepSeek-V4-Flash FP8 원본에 직접 해당한다.

### 대조 대상의 포맷 (중요)
공식 `deepseek-ai/DeepSeek-V4-Flash` 는 config 상 **W8A8 블록 FP8**
(`quant_method: fp8`, `fmt: e4m3`, `weight_block_size: [128,128]`, `scale_fmt: ue8m0`),
148.7 GiB. 우리 llama.cpp 기준선은 `UD-Q4_K_XL`(헤더상 MXFP4) 144.4 GiB 로 **비트폭이 두 배
다르다.** 크기가 비슷한 것은 GGUF 변환본이 파라미터를 284.33B 로 세는 반면 FP8 원본은
그보다 작기 때문이다. **14.59 tok/s 와 22–25 를 그대로 나란히 놓아서는 안 된다.**

## FreeToken end to end (2026-08-23)

Server: `ft serve --model-path /root/models/DeepSeek-V4-Flash --host 0.0.0.0 --port 1919`,
resolved to `moe_backend=offload`, `attention_backend=dsv4_sparse`, `cache_type=swa_radix`,
`page_size=128`. Client: prompt 512 tokens, generate 128, 3 rounds, streaming, decode measured
with TTFT excluded — matched to `llama-bench -p 512 -n 128`.

### The checkpoint is fp4, not fp8

| dtype | bytes | share | what it is |
|---|---|---|---|
| I8 | 132.0 GiB | 88.8% | routed experts, fp4 packed two per byte (283.5B logical params) |
| F8_E8M0 | 8.3 GiB | 5.6% | block scales — MXFP4 family |
| F8_E4M3 | 5.6 GiB | 3.8% | shared experts and dense |
| BF16 | 2.6 GiB | 1.8% | norms, embeddings |

148.6 GiB total against the llama.cpp UD-Q4_K_XL GGUF's 144.44 GiB, whose header also reports
MXFP4 for the MoE. Same format class, 2.9% apart in size — the comparison below is like for like.

### Result and ablation

One variable changed: the MoE cache size. (512 is the floor — below it prefill overlap's two
borrowed expert-layer buffers trip an assertion.)

| configuration | residency | decode peak / steady | TTFT | vs llama.cpp |
|---|---|---|---|---|
| llama.cpp `-ncmoe 40`, static | — | 14.59 | — | baseline |
| FreeToken, `--moe-cache-size 512` | 4.65% | **16.24** / 15.3 | 9.32 s | +11% |
| FreeToken, `--moe-cache-auto` (1195) | 10.9% | **21.48** / 20.3 | 3.15 s | **+47%** |

**76% of the advantage is router-following LRU residency; 24% is the overlapped CPU+PCIe host
path.** The host-bandwidth arithmetic closes at both points. Each token activates 6 experts across
43 layers at 12.8 MiB per expert = **3.21 GiB** if every lookup misses:

- at 4.65% residency, 16.24 tok/s × 3.21 GiB = **52.2 GB/s**, which is the overlapped host ceiling
  measured independently on this box (53.68 GB/s) — so the hit rate is essentially **zero**. Below
  some threshold LRU catches nothing.
- at 10.9% residency, the same ceiling implies only 2.50 GiB read per token → a **22% hit rate,
  twice the residency**. That factor of two is the routing locality the paper is built on.

Raw: `ft_bench_auto.log`, `ft_bench_512.log`, `ft_stats_auto.txt`, `ft_stats_512.txt`.

### What this means for the MLX stack

Nothing, and now for a measured reason. Both halves of FreeToken's advantage are ways of coping
with **VRAM smaller than the model**: the q\* split divides work between a device and a host across
a link, and LRU residency decides which experts are worth keeping on the device. On unified memory
there is no link to divide across and every expert is already resident, so both terms are
identically zero. The paper's claims hold — they were reproduced here at 1.47× llama.cpp on the
hardware they are aimed at — and they are aimed at hardware we do not have.

## Can the iGPU and the NPU help? (2026-08-23)

No. Three configurations, MoE cache pinned at 1195 slots (10.9% residency) so only the backend
differs, same client (512-token prompt, 128 generated, 3 rounds):

| configuration | decode peak | median TTFT | vs baseline |
|---|---|---|---|
| `offload` (default) | **21.39** | 3.35 s | baseline |
| `--moe-backend hybrid` | 13.21 | 3.33 s | **−38.2%** |
| `offload --moe-cpu-layers 8` | 15.79 | 3.25 s | **−26.2%** |

Baseline reproducibility 0.4% (21.39 against 21.48 measured earlier). **TTFT is unchanged across
all three** — this lever touches decode only.

**The default path gives host compute nothing to do.** `--moe-cpu-layers` documents "Unset = all
layers on GPU", so under `offload` the CPU performs *zero* expert arithmetic. Re-deriving the
earlier ablation against a saturated PCIe link instead of a CPU+PCIe overlap makes both residency
points land on the same number — 49.7 GB/s, with implied hit rates of 11.3% and 32.9% — which is
the signature of a link-bound path.

**Moving work to the host is available, and it loses.** GPU utilization falls from 98% to 46–57%
while the CPU rises to ~50%: the work moved, and the GPU now spends half its time waiting. Cutting
the share to 8 of 43 layers still costs 26%, so this is not hybrid's auto-split being mistuned —
host compute loses on its own merits.

**Nothing on this box is decisively faster than the link it would relieve.** Measured expert-delivery
rates: PCIe Gen5 x16 H2D **49.70**, iGPU **53.42**, NPU **48.00**, CPU **36.47** GB/s. The iGPU is
7% above the link, the NPU 3% below it — and all three share one memory controller with the PCIe
DMA, so they cannot be summed against it.

**The physics ceiling is +46%; the measurement is −38%.** PCIe already consumes 49.7 GB/s of DRAM,
leaving 22.6 GB/s of the 72.35 GB/s read ceiling for host compute; with perfect overlap that would
be 72.3 GB/s, or 31.1 tok/s. The 84-point gap between that and what hybrid actually does is
**batch-1 GEMV latency and 43 synchronizations per token**, not bandwidth. Raising host compute
1.61× by adding the iGPU and NPU (36.47 → 58.71 GB/s) scales 13.21 to **21.3 tok/s — a tie with
the 21.39 baseline.** The best case is break-even, and there is no OpenVINO or Level-Zero backend
in the engine, so it would be an implementation rather than a flag.

The link is also already maxed: `nvidia-smi` under load reports **Gen 5 x16** at 98% GPU
utilization (the Gen1 reading at idle is ASPM downtraining, easy to misread), and 49.70 GB/s is 79%
of the 63 GB/s theoretical, which is ordinary overhead.

**The only real lever left on this box is residency, which means VRAM.** Raw: `bench_auto2.log`,
`bench_hybrid.log`, `bench_cpul8.log`, `sweep.log`, `ft_sweep.sh`.
