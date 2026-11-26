"""고유명사 사전: 에이전트 프롬프트와 받아쓰기 hotwords 가 같은 목록을 쓴다.

glossary.json (Mujica 폴더, options.json 옆):
    {
      "토아": ["토하"],                     정확한 표기: [자막에서 잘못 나오는 형태, ...]
      "캔시": [],                           오인식 예시를 모르면 비워 둔다
      "소나": {"variants": ["소나기"],      누구인지 적어 두면 LLM이 역할을 추측하지 않는다
               "note": "게스트로 나온 가수"}
    }

오인식 형태로 자막을 일괄 치환하지 않는다 ("토하다" 같은 일반 단어가 깨진다).
대신 LLM에 알려주고 문맥으로 고치게 한다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_file(path) -> dict[str, dict[str, Any]]:
    """glossary.json 을 읽는다. 파일이 없으면 빈 사전."""
    path = Path(path)
    if not path.exists():
        return {}
    return load({"glossary": json.loads(path.read_text(encoding="utf-8"))})


def load(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """{표기: {"variants": [...], "note": str | None}}"""
    raw = cfg.get("glossary") or {}
    out: dict[str, dict[str, Any]] = {}
    for term, val in raw.items():
        term = str(term).strip()
        if not term:
            continue
        note = None
        if isinstance(val, dict):
            note = (str(val.get("note") or "").strip()) or None
            val = val.get("variants")
        if isinstance(val, str):
            val = [val]
        out[term] = {"variants": [str(v).strip() for v in (val or []) if str(v).strip()], "note": note}
    return out


def hotwords(glossary: dict[str, dict[str, Any]]) -> str | None:
    """Whisper hotwords 문자열. 매 30초 창마다 프롬프트 앞에 붙는다."""
    return ", ".join(glossary) or None


def prompt_block(glossary: dict[str, dict[str, Any]]) -> str:
    """에이전트 프롬프트에 넣을 안내문. 사전이 비면 빈 문자열."""
    if not glossary:
        return ""
    lines = []
    for term, g in glossary.items():
        note = f" — {g['note']}" if g["note"] else ""
        hint = f" (자막에서 {', '.join(repr(v) for v in g['variants'])}(으)로 잘못 나올 수 있음)" if g["variants"] else ""
        lines.append(f"- {term}{note}{hint}")
    return (
        "\n고유명사 사전\n"
        "- 아래 이름·용어는 이 영상에 나올 수 있다. 자막에 비슷한 소리의 단어가 나오면 문맥상 맞는 경우에만 "
        "이 표기로 고쳐 써라. 일반 단어로 쓰인 경우(예: '토하다')는 고치지 마라.\n"
        "- 설명(—)이 없는 이름은 누구인지 추측하지 마라. 특히 영상 속 주인공 본인이라고 단정하지 말고, "
        "자막에 나온 대로 이름만 쓴다.\n"
        + "\n".join(lines)
        + "\n"
    )
