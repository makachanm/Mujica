"""Whisper 자막 사후 필터: 할루시네이션·가사 찌꺼기를 걸러낸다.

세그먼트마다 avg_logprob, no_speech_prob, compression_ratio 가 있어야 제대로 걸러진다.
그래서 받아쓰기 서버에 response_format=verbose_json 으로 요청한다 (addon/whisper_segments.py).

"""
from __future__ import annotations

import re
from typing import Any

# 한국어 Whisper가 무음·음악에서 자주 지어내는 문장. 사람이 실제로 자주 하는 말인 "구독 감사합니다"는 넣지 않는다.
DEFAULT_BLACKLIST = [
    r"시청\s*해\s*주셔서\s*감사합니다",
    r"구독과\s*좋아요",
    r"좋아요와\s*구독",
    r"자막\s*(제공|제작)",
    r"한글\s*자막",
    r"MBC\s*뉴스",
    r"다음\s*(영상|시간)에서?\s*(만나요|뵙겠습니다)",
    r"끝까지\s*시청",
]

# 필터 기준 기본값
DEFAULT_FILTER = {
    "max_compression_ratio": 2.4,  # 같은 말 반복
    "min_avg_logprob": -1.0,       # 신뢰도 낮음 (가사·잡음을 잘못 받아적은 경우가 대부분)
    "max_no_speech_prob": 0.6,     # 무음 확률 높음 (logprob도 낮을 때만)
    "max_repeat": 2,               # 같은 문장이 이보다 많이 연속되면 나머지는 버림
    # "blacklist": [...]           # 비우면 DEFAULT_BLACKLIST
}


def _in_songs(seg: dict[str, Any], songs: list[dict[str, Any]]) -> bool:
    """자막 전체가 무음 처리한 노래 구간 안에 있을 때만 True (걸쳐 있는 코멘트는 살린다)."""
    return any(s["start"] <= seg["start"] and seg["end"] <= s["end"] + 0.5 for s in songs)


def filter_segments(
    segments: list[dict[str, Any]], cfg: dict[str, Any], songs: list[dict[str, Any]] | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(남긴 세그먼트, 버린 세그먼트[+reason])."""
    patterns = [re.compile(p) for p in (cfg.get("blacklist") or DEFAULT_BLACKLIST)]
    max_cr = cfg.get("max_compression_ratio", 2.4)
    min_lp = cfg.get("min_avg_logprob", -1.0)
    max_ns = cfg.get("max_no_speech_prob", 0.6)
    max_repeat = int(cfg.get("max_repeat", 2))
    songs = songs or []

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    run_text, run_len = None, 0
    for s in segments:
        text = s["text"].strip()
        norm = re.sub(r"[\s.,!?~…]+", "", text)
        run_len = run_len + 1 if norm == run_text else 1
        run_text = norm

        reason = None
        if not norm:
            reason = "empty"
        elif _in_songs(s, songs):
            reason = "song"
        elif s.get("compression_ratio", 0) > max_cr:
            reason = "compression_ratio"
        elif s.get("avg_logprob", 0) < min_lp:
            reason = "low_logprob"
        elif s.get("no_speech_prob", 0) > max_ns and s.get("avg_logprob", 0) < -0.5:
            reason = "no_speech"
        elif any(p.search(text) for p in patterns):
            reason = "blacklist"
        elif run_len > max_repeat:
            reason = "repeat"

        if reason:
            dropped.append({**s, "reason": reason})
        else:
            kept.append(s)
    return kept, dropped
