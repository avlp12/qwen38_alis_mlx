# DSv4-Flash TP2 캠페인 핸드오프 (2026-08-23 마감)

정본 원장: `~/qwen38_alis_mlx/docs/LEDGER.md` [I240]–[I264] (AIF 노드 전부 그쪽에 있음 — 이 문서는 상태 스냅숏+운영 부록).

## 정보 노드 (원장 요약)

- [I1] 서빙 최종 스택: **다중-슬롯(bs8) + 프리픽스 스냅숏** `serve_batched_tp2.py` @ :8003, jaccl 메시 1링크(hostfile_jaccl2.json, ips=정적 TB IP 10.0.0.x), MTP depth1. 실측: 디코드 단일 41.1 / 집계 114 tok/s, 프리필 13.9K e2e 544–558 tok/s(투 박스; 싱글 박스 422–473), 멀티턴 TTFT −61%(스냅숏 95% 재사용, greedy 무손실).
- [I2] TB5 멀티링크 캠페인 완결: TCP 링 3케이블 9.6GB/s(2.0×) / **jaccl-ring 3링크 15.46GB/s**(패치 불필요 — 네이티브 4와이어 스트라이핑). 레이턴시 불변 → 디코드 무이득.
- [I3] PR#1189 게시 2건: 본 코멘트(편차 3건: rope/YaRN 배정·풀 마스크 비인과·풀 행 무회전) + 팔로업(풀-rope 수정으로 19K CJK 슬립 0 실증). 로컬 venv_dsv4 빌드는 3편차 전부 수정·검증 완료, 서빙 스택(omlx 오버레이)은 원래 무결.
- [I4] 엡실론 GPU 웨지 사후분석: 8/22 좀비(prefill_2box.server 14.4GB)+6일 업타임 → IOSurfaceSharedEvent 영구 대기. 재부팅 완치. 판별법은 원장 [I240]과 메모리 tb5-multilink 함정 카탈로그.

## 추론/결정 노드

- [RA1] I2+I1: 3링크 서빙 채택은 기각([I258]) — 프리필 연산-지배(+2.6%뿐)·디코드 레이턴시-지배(−7%). 15.46GB/s의 용처는 대역폭-지배 작업(분산 덤프·가중치 동기화)뿐.
- [PA1] 서빙 정본 = 메시 1링크 유지. 3링크 구성은 `hostfile_jaccl_r3.json` 보존, 필요 작업에서만 투입.
- [PA2] 열린 작업 없음. 대기성: PR#1189 업스트림 반응 모니터링(코멘트 2건), jaccl 멀티-rdma를 실제 대역폭-지배 작업에 투입해 보는 것(옵션).

## 부록 (운영 데이터)

```
서빙 재기동:
  cd /Users/Shared/tp2 && nohup ~/venv_omlx063/bin/mlx.launch \
    --hostfile hostfile_jaccl2.json /Users/Shared/tp2/serve_b.sh > serve_b.log 2>&1 &
  READY 판정: serve_b.log 의 "서빙 시작" 라인 (포트 점유로 판정 금지 — 고스트 전례)
  단일-슬롯 예비: serve.sh (serve_tp4_dspark.py, control-port 18003)

핵심 경로:
  모델 팩:        ~/dsv4flash/mlx4bit (MTP 복원 포함; 스톡 로더용 뷰 mlx4bit_nomtp)
  수정 mlx-lm:    ~/venv_dsv4/.../mlx_lm/models/deepseek_v4.py (3편차 수정본,
                  사본 ~/qwen38_alis_mlx/results/exp20_dsv4_mac/deepseek_v4_patched.py)
  검증 하네스:    ~/qwen38/exp19_ft_mods/rope_pool_test.py (CJK 슬립 계수 포함)
  멀티링크 로그:  ~/local_claude_code/tb5_bench/multilink_ips{1,2,3}.log
  jaccl 벤치:     /Users/Shared/tb5_bench/{jbench.py,jb.sh,hj1.json,hj3.json}

네트워크(재부팅 안전):
  TB브리지 서비스 양 머신 영구 off / com.alis.tbnet 데몬이 부팅 시 TB IP 복원
  배선: en4↔en4=10.0.0.x, en5↔en5=10.0.1.x, en3↔en3=10.0.2.x (g=.1, e=.2)
  epsilon ssh: m3ms@10.0.0.2 · 게지히트 LAN IP는 DHCP 유동(.154였음) — 정적 참조 금지

게시물:
  PR#1189 본 코멘트: github.com/ml-explore/mlx-lm/pull/1189#issuecomment-5386017983
  PR#1189 팔로업:   github.com/ml-explore/mlx-lm/pull/1189#issuecomment-5386141413
```
