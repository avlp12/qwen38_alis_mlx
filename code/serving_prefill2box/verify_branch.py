"""길이 분기 라이브 검증 — 임계값 아래/위 요청을 하나씩 보내고
서버 로그가 실제로 다른 청크를 골랐는지 확인한다.

프리픽스 캐시가 재사용되면 프리필 자체가 일어나지 않아 분기를 관측할 수 없으므로
요청마다 고유 접두를 붙인다([I152] 계열 함정).
"""
import json, sys, time, urllib.request, uuid

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8004
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

TOK_PER_WORD = 3.89   # 이 코퍼스 실측 (16341 토큰 / 4200 단어)


def ask(target_tokens, label):
    n_words = int(target_tokens / TOK_PER_WORD)
    # 고유 접두 — 프리픽스 캐시 재사용 차단
    body = f"[{uuid.uuid4()}] " + " ".join(
        f"item{i%997}" for i in range(n_words)
    )
    req = urllib.request.Request(
        URL,
        data=json.dumps({
            "messages": [{"role": "user", "content":
                          body + "\n\nReply with exactly one word: ok"}],
            "max_tokens": 4, "temperature": 0.0,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    dt = time.perf_counter() - t0
    u = d.get("usage", {})
    print(f"[{label}] prompt_tokens={u.get('prompt_tokens')} "
          f"ttft+gen={dt:.2f}s", flush=True)
    return u.get("prompt_tokens")

# min_tokens 4096 아래(단일 박스) / 그 위·임계값 아래(좁은 청크) / 임계값 위(넓은 청크)
ask(3200, "단일박스")
ask(8000, "좁은-청크")
ask(13000, "넓은-청크")
print("VERIFY-DONE", flush=True)
