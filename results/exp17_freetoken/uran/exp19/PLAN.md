# exp19 — FreeToken 엔진 개조 캠페인: MTP 자기-스펙 + 프리필 호스트-어시스트

지시(2026-08-23): "둘 다 진행". 대상 엔진 freetoken 0.1.2 (uran WSL2, 순수 Python+Triton,
소스 회수·git 추적: `freetoken_src/`, pristine 태그 21dd595).

## 정찰 확정 사실

- **[I301]** 체크포인트에 MTP 완전체 존재: `mtp.0.*` 1,575텐서 = MLA형 어텐션(wq_a/b·wkv·
  wo_a/b·attn_sink·q_norm/kv_norm) + enorm/e_proj + **자체 256-전문가 MoE FFN**(fp4+scale).
  엔진은 명시적으로 스킵(weight.py:205,319) — n_mtp_layers=1 은 args 에만 존재.
- **[I302]** 라우터 구조: 층 0-2 는 **해시 라우팅**(`tid2eid[input_ids]` — 토큰 id 의 순수
  함수 → 드래프트 시점에 전문가 확정 = 완전 프리페치 가능), 층 3-42 는 sqrtsoftplus top-6.
- **[I303]** 디코드는 CUDA-graph 캡처(`GraphRunner`), `--cuda-graph-max-bs 0` = eager.
  프리필은 eager ragged 배치(`prefill_batched`, cu_seqlens, radix 연속 extend 지원,
  start_pos>0 재개 가능) — **검증(verify)을 2-토큰 extend 로 태울 자연 경로**.
- **[I304]** DSV4 `_prefill_routed` 오버라이드: `T*top_k < num_experts` 면 디코드식
  슬롯-캐시 경로(ensure_experts+copy_missing) → **2-토큰 verify 의 전문가 합집합 dedup 은
  슬롯 캐시가 공짜로 제공**.
- **[I305]** 프리필 본경로는 층 전체(256전문가) 이중버퍼 스트리밍(prefetch L, L+1 → wait →
  GEMM → release). TTFT 3.2s 의 약 75%가 이 스트리밍(117.6 GiB ÷ 49.7 GB/s ≈ 2.4s).
- **[I306]** hybrid 의 GPU/CPU 분할·병합 패턴 재사용 가능: `decode_submit`(-1 id 스킵,
  pinned IO per-bs, 플래그-싱크) + GPU 쪽 가중치 제로잉 병합. CPU 실행기는 임의 bs 수용
  (`_io_for(bs)`), IO 버퍼는 bs 별 공유(층 무관).

## Track P — 프리필 호스트-어시스트 (선행)

설계: `_prefill_routed` 오버랩 경로에서 층당 256전문가를 (GPU-스트림, CPU-계산) 으로 분할.
CPU 몫 전문가는 스트리밍에서 제외(H2D 바이트 절감)하고 해당 라우트를 cpu_executor 로
제출(라우트 id 유지, GPU 쪽은 해당 라우트 가중치 0). N_cpu 는 환경변수 스윕(0/16/32/48).
- 예상: TTFT −8~15% (CPU GEMM 이 dequant-바운드면 하한, L3 상주 GEMM 이면 상한).
- 위험: 대형 bs pinned 버퍼(버킷 패딩으로 통제), C++ 커널의 대 bs 거동 미지(스윕 0 폴백).

## Track M — MTP-1 자기-스펙 (계측 후)

게이트 측정(진행 중): 연속-토큰 top-k 겹침. 이득 ≈ (1+α)/(2−ov)−1 —
ov=0.3·α=0.8 → +6%, ov=0.5·α=0.85 → +23%. bs=4 동률 실측이 "스텝 비용 = 토큰 선형"을
보였으므로 **ov 가 낮으면 MTP 는 산술적으로 죽는다** — 그래서 계측이 먼저다.
설계 스케치(측정 통과 시): MTP 모듈(자체 MLA state + 공유 embed/head, FFN 은 공유
오프로드 캐시 경유) · verify = 2-토큰 radix-extend(프리필 경로) · greedy 수락 ·
거부 시 상태 롤백(컴프레서 carry 스냅숏) · eager-extend 스텝 비용 측정 선행.
- 위험(대): DSA 상태 롤백, eager 43층 launch 오버헤드, MTP 어텐션의 정확 구조 미상
  (참조 구현 필요 — DeepSeek V4 reference inference/model.py 대조).

## 안전 규약

패치는 로컬 git 브랜치 → scp 로 site-packages 반영 → 실험 → **pristine 복원 스크립트**
(`git show 21dd595:<f>` 로 원본 재배포). 서버는 실험 후 항상 종료(서빙 복구는 지시 시).

## 진행 로그 (2026-08-23)

- **[I307]** 라우팅 계측 완료(1,008스텝×43층): 인접 겹침 44.6%(비해시 47.7%),
  8-창 68.0%, 해시층 2.3%. eager 디코드 ≈7.3 tok/s(계측 포함) → 그래프 가치 약 3배.
- **[I308]** LRU 정밀 시뮬 → 비용 모델 확정: 스텝 = 17.1ms + 0.38ms×미스.
  "링크 포화" 철회([CA49], 게시 정정 완료). k5/k4 실측이 시뮬 예측 미달 —
  k-레버는 엔진 수술 없이 닫힘([I226]).
- **[RA80-P]** Track M 산술 갱신: 이득 채널은 인출 dedup(LRU가 이미 흡수)이 아니라
  **고정 17.1ms의 (1+α) 상각**. 예상 +5~12%(α 0.7~0.85), 단 verify 는 그래프 필수
  (eager 는 스텝당 약 +90ms 로 즉사). → α 실측(오프라인 하네스)로 게이트.
- Track P 1차 스윕: 배포 누락 사고(패치가 site-packages 에 안 감 — 배포 검증 grep 을
  스크립트에 내장할 것). 부수확: 기준선 TTFT 재현성 3.13/3.15s. 스윕2 재발사(검증 완료).
