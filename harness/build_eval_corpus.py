"""평가 코퍼스 생성 — ppl_eval.py 가 먹는 3개 슬라이스(en/ko/code)를 만든다.

왜 새로 만드나: 카드에 실린 top-1 수치는 62~112토큰 프로브라 1토큰이 1pp 를 넘는다.
코퍼스 규모(수만 토큰)로 갈아타야 티어 간·레시피 간 차이를 통계로 말할 수 있다.

출처(전부 로컬, 재현 가능):
  en   = wikitext-2-raw-v1 **test** split (HF 캐시 arrow, Salesforce/wikitext)
  ko   = 한국어 위키 원문 덤프 kimi_k3/phase2/ko_pretrain/kowiki_raw.txt 의
         중간 오프셋 슬라이스 (앞머리는 최다-인용 문서라 암기 편향이 있어 피한다)
  code = CPython 표준 라이브러리 소스 묶음 (버전 고정 없이도 재현 가능한 공개 코드)

오염 주의: AWQ 캘리브레이션은 ~/.cache/mlx-lm/calibration_v5.txt 를 썼다(437KB).
위 3개 슬라이스 중 어느 것도 그 파일이 아니다. 다만 calibration_v5 는 잡다한
위키·코드·다국어 혼합이라 주제 수준의 겹침은 배제할 수 없다. 절대 PPL 이 아니라
같은 코퍼스 위에서의 빌드 간 상대 비교가 목적이므로 판정에는 영향이 없다.
"""
import os
import sys

OUT = "/Users/gesicht/qwen38/eval_corpus"
os.makedirs(OUT, exist_ok=True)

# ── en: wikitext-2-raw-v1 test ────────────────────────────────────────────
EN_ARROW = ("/Users/gesicht/.cache/huggingface/datasets/Salesforce___wikitext/"
            "wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/"
            "wikitext-test.arrow")


def build_en(target_chars):
    import pyarrow as pa
    with pa.memory_map(EN_ARROW, "r") as src:
        tbl = pa.ipc.open_stream(src).read_all()
    txt = "".join(tbl.column("text").to_pylist())
    print(f"[en] wikitext-2 test 전체 {len(txt)} chars", file=sys.stderr)
    return txt[:target_chars]


# ── ko: 한국어 위키 ────────────────────────────────────────────────────────
KO_RAW = "/Users/gesicht/kimi_k3/phase2/ko_pretrain/kowiki_raw.txt"
KO_OFFSET = 40_000_000          # 덤프 중간 — 앞머리 유명 문서 회피


def build_ko(target_chars):
    with open(KO_RAW, encoding="utf-8") as f:
        f.seek(KO_OFFSET)
        f.readline()             # 깨진 첫 줄 버림
        buf = f.read(target_chars * 3)
    # 너무 짧은 줄(제목·표 파편)은 산문 PPL 을 흔들므로 제거
    lines = [l for l in buf.split("\n") if len(l.strip()) >= 40]
    return "\n".join(lines)[:target_chars]


# ── code: CPython 표준 라이브러리 ──────────────────────────────────────────
STDLIB_FILES = [
    "dataclasses.py", "argparse.py", "json/encoder.py", "json/decoder.py",
    "difflib.py", "fractions.py", "statistics.py", "textwrap.py",
    "shutil.py", "ipaddress.py", "csv.py", "configparser.py",
    "http/cookiejar.py", "email/message.py", "asyncio/tasks.py",
    "asyncio/streams.py", "typing.py", "enum.py", "functools.py",
    "pathlib/_local.py", "pathlib.py", "random.py", "heapq.py", "bisect.py",
    "collections/__init__.py", "contextlib.py", "inspect.py", "logging/__init__.py",
]


def build_code(target_chars):
    import sysconfig
    root = sysconfig.get_paths()["stdlib"]
    # 파일당 상한을 둔다 — 안 두면 dataclasses+argparse 둘로 예산이 다 차서
    # "코드" 슬라이스가 사실상 두 파일의 문체 측정이 된다(실제로 그랬다).
    per = max(4000, target_chars // max(1, len(STDLIB_FILES)))
    parts, seen = [], 0
    for rel in STDLIB_FILES:
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8", errors="ignore") as f:
            s = f.read()[:per]
        parts.append(f"# === {rel} ===\n{s}\n")
        seen += len(s)
        if seen >= target_chars:
            break
    print(f"[code] stdlib root={root} 파일 {len(parts)}개 (파일당 ≤{per} chars)",
          file=sys.stderr)
    return "".join(parts)[:target_chars]


if __name__ == "__main__":
    # 문자 예산은 아래 토크나이저 실측으로 정한 값(슬라이스당 ~35k 토큰 목표):
    #   en   ~3.9 char/tok, ko ~1.6 char/tok, code ~3.3 char/tok
    for name, fn, n in (("en", build_en, 150_000),
                        ("ko", build_ko, 60_000),
                        ("code", build_code, 130_000)):
        s = fn(n)
        p = os.path.join(OUT, f"{name}.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write(s)
        print(f"{name}: {len(s)} chars → {p}")
