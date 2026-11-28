"""
Mujica 평가. cases.json 의 사례마다 파이프라인을 여러 번 돌려 정답 구간과 비교한다.

    .venv/bin/python -m benchmark.run_mujica --runs 5
    .venv/bin/python -m benchmark.run_mujica --runs 5 --set agent_model=claude-haiku-5-5 --set vision_model=claude-haiku-5-5
    .venv/bin/python -m benchmark.run_mujica --case bbb_bow_making --set chunk_count=4

설정은 options.json(화면에서 저장한 값)에서 시작해, 사례의 language 와 --set 값으로 덮어쓴다.
결과는 benchmark/results/<시각>_mujica_<모델>/ 에 남는다. 후보 클립은 남기지 않는다.
"""

import argparse
import asyncio
import tempfile
import time
from datetime import datetime

from benchmark.scoring import (
    COLUMNS, add_tokens, build_table, code_version, load_cases, new_result_dir, print_run, save_report, score,
    video_path,
)
from main import find_moments
from options import DEFAULT_OPTIONS, load_options


def parse_overrides(pairs):
    """--set 이름=값 들을 옵션 값으로 바꾼다. 값의 형은 기본값의 형을 따른다."""
    overrides = {}
    for pair in pairs:
        name, value = pair.split("=", 1)
        if name not in DEFAULT_OPTIONS:
            raise SystemExit(f"모르는 옵션입니다: {name} (가능한 옵션: {', '.join(DEFAULT_OPTIONS)})")

        default = DEFAULT_OPTIONS[name]
        if isinstance(default, bool):
            overrides[name] = value.lower() == "true"
            continue
        if isinstance(default, int):
            overrides[name] = int(value)
            continue
        if isinstance(default, float):
            overrides[name] = float(value)
            continue
        overrides[name] = value
    return overrides


async def run_case(case, settings):
    async def report_progress(description, done):
        pass

    started_at = time.time()
    with tempfile.TemporaryDirectory(prefix="mujica_clips_") as clip_dir:
        segments, tokens = await find_moments(
            case["request"], case["video_description"], video_path(case), settings, clip_dir, report_progress,
        )
    found = [[segment.start, segment.end] for segment in segments]
    result = score(found, case["answers"])
    result["seconds"] = time.time() - started_at
    add_tokens(result, tokens)
    result["found"] = found
    result["reasons"] = [segment.reasons for segment in segments]
    return result


def summarize(cases, results, settings, overrides, version):
    lines = [
        f"# Mujica 평가 ({datetime.now():%Y-%m-%d %H:%M})",
        "",
        f"- 코드: `{version}`",
        f"- 에이전트 모델: `{settings['agent_model']}`, 비전 모델: `{settings['vision_model']}`",
        f"- 조각 {settings['chunk_count']}개, 윈도우 {settings['window_seconds']}초, 한 번에 {settings['frames_per_look']}장",
        f"- Whisper: `{settings['whisper_model']}`, 노래 감지: {settings['detect_songs']}",
        f"- 덮어쓴 값: {overrides if overrides else '없음'}",
        "",
    ]
    return "\n".join(lines + build_table(cases, results, COLUMNS)) + "\n"


async def main():
    parser = argparse.ArgumentParser(description="cases.json 의 사례로 Mujica 를 평가한다.")
    parser.add_argument("--runs", type=int, default=1, help="사례마다 몇 번 돌릴지")
    parser.add_argument("--set", action="append", default=[], metavar="이름=값", help="옵션 덮어쓰기 (여러 번 가능)")
    parser.add_argument("--case", action="append", default=[], metavar="이름", help="이 사례만 돌린다 (여러 번 가능)")
    parser.add_argument("--name", default="", help="결과 폴더 이름 (기본: 시각_mujica_모델)")
    arguments = parser.parse_args()

    cases = load_cases(arguments.case)
    overrides = parse_overrides(arguments.set)
    base_settings = load_options()
    base_settings.update(overrides)
    result_dir = new_result_dir("mujica", base_settings["agent_model"], arguments.name)

    results = []
    for case in cases:
        settings = dict(base_settings)
        settings["language"] = case["language"]
        for run_number in range(1, arguments.runs + 1):
            # 서버 오류 같은 실패도 기록하고 다음 실행으로 넘어간다. 실패한 실행은 평균에서 빠진다.
            try:
                result = await run_case(case, settings)
            except Exception as error:
                result = {"error": f"{type(error).__name__}: {error}"}
            result["case"] = case["name"]
            result["run"] = run_number
            results.append(result)
            print_run(result)

    version = code_version()
    # 언어 코드는 사례마다 다르므로 cases 쪽에 있다. API 키와 서버 주소는 기록에 남기지 않는다.
    base_settings["anthropic_api_key"] = ""
    base_settings["ollama_url"] = ""
    base_settings["whisper_server_url"] = ""
    report = {"code_version": version, "settings": base_settings, "overrides": overrides, "cases": cases, "results": results}
    save_report(result_dir, report, summarize(cases, results, base_settings, overrides, version))


if __name__ == "__main__":
    asyncio.run(main())
