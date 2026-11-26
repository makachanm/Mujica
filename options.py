"""화면 옵션의 기본값과, 저장된 옵션 파일(options.json) 읽기/쓰기."""

import json
from pathlib import Path


OPTIONS_PATH = Path(__file__).parent / "options.json"

# 이 순서가 화면의 옵션 칸 순서와 같아야 한다 (ui.build_options_panel 참고).
DEFAULT_OPTIONS = {
    "ollama_url": "http://localhost:11434",
    "agent_model": "qwen3:8b",
    "vision_model": "qwen2.5vl:7b",
    "chunk_count": 4,
    "window_seconds": 10.0,
    "frames_per_look": 8,
    "whisper_server_url": "http://localhost:8000",
    "whisper_model": "Systran/faster-whisper-small",
    "language": "ko",
    "detect_songs": True,
}


def load_options():
    """저장된 값이 있으면 그 값을, 없는 항목은 기본값을 쓴다."""
    options = dict(DEFAULT_OPTIONS)
    if not OPTIONS_PATH.exists():
        return options

    saved = json.loads(OPTIONS_PATH.read_text(encoding="utf-8"))
    for name in DEFAULT_OPTIONS:
        if name in saved:
            options[name] = saved[name]
    return options


def save_options(options):
    OPTIONS_PATH.write_text(json.dumps(options, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
