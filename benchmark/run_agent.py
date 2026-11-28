"""
기준선 평가. Mujica 파이프라인 대신 범용 에이전트(Claude Code)에게 같은 요청을 주고 속도와 정확도를 잰다.

    .venv/bin/python -m benchmark.run_agent --runs 5
    .venv/bin/python -m benchmark.run_agent --model claude-haiku-5-5 --case bbb_bow_making --case bbb_spike_trap
    .venv/bin/python -m benchmark.run_agent --model claude-haiku-5-5 --no-whisper

사례마다 빈 임시 폴더를 만들어 영상(video.<확장자>)과 받아쓰기 도구(transcribe.py)만 넣고 그 안에서 Claude Code 를 돌린다.
에이전트는 찾은 장면을 직접 잘라 clips/ 에 저장해야 한다. 프롬프트에는 받아쓰기 도구만 알려 주고,
영상을 보거나 자르는 방법(ffmpeg, 이미지 읽기 등)은 에이전트가 스스로 찾게 둔다.
--no-whisper 를 주면 받아쓰기 도구도 넣지 않고, 프롬프트에서도 도구 이야기를 뺀다.
정답 파일이 있는 레포는 보이지 않는다. 훅, 메모리, CLAUDE.md 같은 개인 설정도 쓰지 않는다(--bare).
받아쓰기 도구는 Mujica 와 같은 Whisper 서버와 모델을 쓴다. API 키는 options.json 의 키를 쓴다.

결과는 benchmark/results/<시각>_agent_<모델>/ 에 남는다. 지표는 run_mujica.py 와 같고, 비용과 턴 수,
에이전트가 만든 클립 수를 더 기록한다. 클립 파일은 clips/<사례>_<실행>/ 에 복사해 둔다(git 제외).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

from common import run_command
from benchmark.scoring import (
    COLUMNS, add_tokens, build_table, code_version, load_cases, new_result_dir, print_run, save_report, score,
    video_path,
)
from options import load_options


DEFAULT_MODEL = "claude-sonnet-5-5"
TIME_LIMIT_SECONDS = 600
BUDGET_PER_RUN_USD = 3.0
AGENT_COLUMNS = COLUMNS + [("clip_count", "클립 수"), ("cost_usd", "비용($)"), ("turns", "턴 수")]

# 임시 폴더에 넣어 주는 받아쓰기 도구. 표준 라이브러리와 ffmpeg 만 쓴다.
TRANSCRIBE_TOOL = '''\
"""사용법: python3 transcribe.py <시작 초> <끝 초>
video 의 [시작, 끝) 구간을 받아써서, 영상 전체 기준 시각과 함께 한 줄씩 출력한다."""

import json
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ENDPOINT = {endpoint!r}
MODEL = {model!r}
LANGUAGE = {language!r}
VIDEO = {video!r}

start = float(sys.argv[1])
end = float(sys.argv[2])
with tempfile.TemporaryDirectory() as folder:
    audio_path = Path(folder) / "audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-t", str(end - start),
         "-i", VIDEO, "-vn", "-ac", "1", "-ar", "16000", str(audio_path)],
        check=True,
    )
    fields = {{
        "model": MODEL, "language": LANGUAGE, "response_format": "verbose_json",
        "timestamp_granularities[]": "segment", "temperature": "0", "without_timestamps": "false",
    }}
    boundary = uuid.uuid4().hex
    body = b""
    for name, value in fields.items():
        body += f"--{{boundary}}\\r\\nContent-Disposition: form-data; name=\\"{{name}}\\"\\r\\n\\r\\n{{value}}\\r\\n".encode()
    body += f"--{{boundary}}\\r\\nContent-Disposition: form-data; name=\\"file\\"; filename=\\"audio.wav\\"\\r\\n".encode()
    body += b"Content-Type: audio/wav\\r\\n\\r\\n" + audio_path.read_bytes() + f"\\r\\n--{{boundary}}--\\r\\n".encode()
    request = urllib.request.Request(ENDPOINT, data=body, headers={{"Content-Type": f"multipart/form-data; boundary={{boundary}}"}})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            segments = json.loads(response.read())["segments"]
    except urllib.error.HTTPError as error:
        print(f"받아쓰기 실패 ({{error.code}}). 말소리가 없는 구간이면 서버가 실패할 수 있다. 더 짧게 나눠 보라.")
        sys.exit(1)

for segment in segments:
    print(f"{{start + segment['start']:.1f}}-{{start + segment['end']:.1f}} {{segment['text'].strip()}}")
'''

PROMPT = """\
이 폴더의 {video} 영상에서 아래 요청에 맞는 장면을 찾아, 장면마다 잘라서 clips 폴더에 clip_1.mp4, clip_2.mp4, ... 로 저장해 주세요.

요청: {request}
{tool_note}
요청에 맞는 장면이 여러 군데면 모두 찾고, 장면이 실제로 시작하고 끝나는 시각에 맞춰 잘라 주세요.
다 저장한 뒤 맨 마지막 줄에는 다른 말 없이, 잘라 낸 구간을 아래 형식의 JSON 한 줄로 출력하세요. 찾지 못했으면 빈 목록을 쓰세요.
{{"segments": [[시작 초, 끝 초], ...]}}
"""

WHISPER_NOTE = """
이 폴더에는 받아쓰기 도구가 있습니다.
- python3 transcribe.py <시작 초> <끝 초> 로 그 구간의 음성을 받아쓸 수 있습니다. 시각은 영상 전체 기준이고, 한 줄에 "시작-끝 문장" 형식으로 출력됩니다.
"""


def prepare_folder(case, settings, folder, with_whisper):
    video_name = "video" + Path(case["video"]).suffix
    shutil.copyfile(video_path(case), Path(folder) / video_name)
    (Path(folder) / "clips").mkdir()
    if not with_whisper:
        return video_name

    tool = TRANSCRIBE_TOOL.format(
        endpoint=settings["whisper_server_url"].rstrip("/") + "/v1/audio/transcriptions",
        model=settings["whisper_model"],
        language=case["language"],
        video=video_name,
    )
    (Path(folder) / "transcribe.py").write_text(tool, encoding="utf-8")
    return video_name


def collect_clips(folder, keep_dir):
    """에이전트가 clips/ 에 만든 파일을 keep_dir 로 복사하고, 파일마다 길이(초)를 잰다. 재생할 수 없으면 None."""
    clips = []
    for clip_path in sorted((Path(folder) / "clips").iterdir()):
        keep_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(clip_path, keep_dir / clip_path.name)
        try:
            length = float(run_command([
                "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(clip_path),
            ]).strip())
        except (RuntimeError, ValueError):
            length = None
        clips.append({"name": clip_path.name, "seconds": length})
    return clips


def read_segments(answer_text):
    """에이전트 답의 마지막 {"segments": ...} JSON 을 읽는다. 없으면 None."""
    matches = re.findall(r'\{\s*"segments"\s*:\s*\[.*?\]\s*\}', answer_text, re.DOTALL)
    if len(matches) == 0:
        return None
    segments = json.loads(matches[-1])["segments"]
    return [[float(start), float(end)] for start, end in segments]


def run_case(case, settings, model, with_whisper, keep_dir):
    with tempfile.TemporaryDirectory(prefix="agent_eval_") as folder:
        video_name = prepare_folder(case, settings, folder, with_whisper)
        tool_note = ""
        if with_whisper:
            tool_note = WHISPER_NOTE
        prompt = PROMPT.format(video=video_name, request=case["request"], tool_note=tool_note)
        environment = dict(os.environ)
        environment["ANTHROPIC_API_KEY"] = settings["anthropic_api_key"]

        started_at = time.time()
        try:
            finished = subprocess.run(
                [
                    "claude", "-p", prompt,
                    "--model", model,
                    "--bare",
                    "--output-format", "json",
                    "--allowedTools", "Bash", "Read",
                    "--permission-mode", "dontAsk",
                    "--max-budget-usd", str(BUDGET_PER_RUN_USD),
                    "--no-session-persistence",
                ],
                cwd=folder, env=environment, capture_output=True, text=True, timeout=TIME_LIMIT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return {"error": f"{TIME_LIMIT_SECONDS}초 시간 제한을 넘김", "seconds": time.time() - started_at}
        seconds = time.time() - started_at
        clips = collect_clips(folder, keep_dir)

    if finished.returncode != 0:
        return {"error": f"claude 종료 코드 {finished.returncode}: {finished.stderr.strip()[:500]}", "seconds": seconds}

    output = json.loads(finished.stdout)
    answer_text = output.get("result", "")
    found = read_segments(answer_text)
    if found is None:
        return {"error": "마지막 줄에 segments JSON 이 없음", "seconds": seconds, "answer": answer_text[-2000:]}

    result = score(found, case["answers"])
    result["seconds"] = seconds
    usage = output.get("usage", {})
    tokens = {
        "input": usage.get("input_tokens", 0),
        "output": usage.get("output_tokens", 0),
        "cache_write": usage.get("cache_creation_input_tokens", 0),
        "cache_read": usage.get("cache_read_input_tokens", 0),
    }
    add_tokens(result, tokens)
    result["cost_usd"] = output.get("total_cost_usd", 0.0)
    result["turns"] = output.get("num_turns", 0)
    result["clip_count"] = len(clips)
    result["clips"] = clips
    result["found"] = found
    result["answer"] = answer_text[-2000:]
    return result


def summarize(cases, results, model, with_whisper, version):
    lines = [
        f"# 범용 에이전트 기준선 ({datetime.now():%Y-%m-%d %H:%M})",
        "",
        f"- 코드: `{version}`",
        f"- Claude Code (--bare), 모델 `{model}`, 도구 Bash·Read, 실행당 {TIME_LIMIT_SECONDS}초·${BUDGET_PER_RUN_USD} 제한",
        f"- 받아쓰기 도구: {'있음' if with_whisper else '없음'}",
        "",
    ]
    return "\n".join(lines + build_table(cases, results, AGENT_COLUMNS)) + "\n"


def main():
    parser = argparse.ArgumentParser(description="범용 에이전트(Claude Code)로 같은 사례를 풀게 해 기준선을 잰다.")
    parser.add_argument("--runs", type=int, default=1, help="사례마다 몇 번 돌릴지")
    parser.add_argument("--case", action="append", default=[], metavar="이름", help="이 사례만 돌린다 (여러 번 가능)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Claude Code 에 물릴 모델")
    parser.add_argument("--no-whisper", action="store_true", help="받아쓰기 도구 없이 돌린다")
    parser.add_argument("--name", default="", help="결과 폴더 이름 (기본: 시각_agent_모델)")
    arguments = parser.parse_args()

    cases = load_cases(arguments.case)
    settings = load_options()
    if settings["anthropic_api_key"] == "":
        raise SystemExit("options.json 에 Claude API 키가 없습니다.")
    with_whisper = not arguments.no_whisper
    result_dir = new_result_dir("agent", arguments.model, arguments.name)

    results = []
    for case in cases:
        for run_number in range(1, arguments.runs + 1):
            keep_dir = result_dir / "clips" / f"{case['name']}_{run_number}"
            result = run_case(case, settings, arguments.model, with_whisper, keep_dir)
            result["case"] = case["name"]
            result["run"] = run_number
            results.append(result)
            print_run(result)

    version = code_version()
    report = {"code_version": version, "agent_model": arguments.model, "with_whisper": with_whisper, "cases": cases, "results": results}
    save_report(result_dir, report, summarize(cases, results, arguments.model, with_whisper, version))


if __name__ == "__main__":
    main()
