"""
두 평가 스크립트(run_mujica.py, run_agent.py)가 같이 쓰는 것: 사례 읽기, 채점, 요약표, 기록 저장.

지표
    tIoU           찾은 구간 전체와 정답 구간 전체가 겹친 길이 / 합친 길이
    정답률         recall@0.5. 정답 구간 가운데, IoU 0.5 이상으로 겹치는 찾은 구간이 있는 비율
    precision@0.5  찾은 구간 가운데, IoU 0.5 이상으로 겹치는 정답 구간이 있는 비율 (찾은 구간이 없으면 0)
    시간           요청을 넣고 답이 나올 때까지 걸린 시간
    입력 토큰      Claude 에 보낸 토큰 (캐시 쓰기·읽기 포함). Ollama 모델은 세지 않는다
    출력 토큰      Claude 가 만든 토큰 (생각 토큰 포함)
"""

import json
import statistics
from datetime import datetime
from pathlib import Path

from common import run_command


BENCHMARK_DIR = Path(__file__).parent
ROOT = BENCHMARK_DIR.parent
CASES_PATH = BENCHMARK_DIR / "cases.json"
RESULTS_DIR = BENCHMARK_DIR / "results"
MATCH_IOU = 0.5

# 요약표의 공통 열: (결과 키, 머리글)
COLUMNS = [
    ("tiou", "tIoU"),
    ("recall", "정답률 (recall@0.5)"),
    ("precision", "precision@0.5"),
    ("segment_count", "찾은 구간 수"),
    ("seconds", "시간(초)"),
    ("input_tokens", "입력 토큰"),
    ("output_tokens", "출력 토큰"),
]


def load_cases(names):
    """cases.json 의 사례. names 가 비어 있지 않으면 그 이름의 사례만 돌려준다."""
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if len(names) == 0:
        return cases
    unknown = set(names) - {case["name"] for case in cases}
    if len(unknown) > 0:
        raise SystemExit(f"cases.json 에 없는 사례입니다: {', '.join(sorted(unknown))}")
    return [case for case in cases if case["name"] in names]


def video_path(case):
    return str(BENCHMARK_DIR / case["video"])


def code_version():
    """지금 코드의 커밋. 커밋하지 않은 변경이 있으면 뒤에 +dirty 를 붙인다."""
    commit = run_command(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]).strip()
    changes = run_command(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"]).strip()
    if changes != "":
        return commit + "+dirty"
    return commit


def merge_ranges(ranges):
    merged = []
    for start, end in sorted(ranges):
        if len(merged) > 0:
            if start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
                continue
        merged.append([start, end])
    return merged


def overlap_length(first_ranges, second_ranges):
    total = 0.0
    for first_start, first_end in merge_ranges(first_ranges):
        for second_start, second_end in merge_ranges(second_ranges):
            total += max(0.0, min(first_end, second_end) - max(first_start, second_start))
    return total


def total_length(ranges):
    return sum(end - start for start, end in merge_ranges(ranges))


def iou(first_ranges, second_ranges):
    overlap = overlap_length(first_ranges, second_ranges)
    union = total_length(first_ranges) + total_length(second_ranges) - overlap
    if union <= 0:
        return 0.0
    return overlap / union


def count_matched(ranges, others):
    """ranges 가운데 others 의 어느 한 구간과 IoU 가 MATCH_IOU 이상인 것의 수."""
    matched = 0
    for single in ranges:
        best = max((iou([single], [other]) for other in others), default=0.0)
        if best >= MATCH_IOU:
            matched += 1
    return matched


def score(found, answers):
    precision = 0.0
    if len(found) > 0:
        precision = count_matched(found, answers) / len(found)
    return {
        "tiou": iou(found, answers),
        "recall": count_matched(answers, found) / len(answers),
        "precision": precision,
        "segment_count": len(found),
    }


def add_tokens(result, tokens):
    """tokens = {input, output, cache_write, cache_read} 를 결과에 적는다."""
    result["input_tokens"] = tokens["input"] + tokens["cache_write"] + tokens["cache_read"]
    result["output_tokens"] = tokens["output"]
    result["tokens"] = tokens


def build_table(cases, results, columns):
    """사례별 평균 ± 표준편차 표. 실패한 실행은 평균에서 빠진다."""
    headers = ["사례", "성공/실행"] + [header for key, header in columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for case in cases:
        runs = [result for result in results if result["case"] == case["name"]]
        finished = [run for run in runs if "error" not in run]
        cells = []
        for key, header in columns:
            values = [run[key] for run in finished]
            if len(values) == 0:
                cells.append("-")
                continue
            spread = 0.0
            if len(values) > 1:
                spread = statistics.stdev(values)
            cells.append(format_mean(statistics.mean(values), spread))
        lines.append(f"| {case['name']} | {len(finished)}/{len(runs)} | " + " | ".join(cells) + " |")
    return lines


def format_mean(mean, spread):
    # 토큰처럼 큰 수는 소수점 없이 천 단위로 끊어 쓴다.
    if mean >= 1000:
        return f"{mean:,.0f} ± {spread:,.0f}"
    return f"{mean:.2f} ± {spread:.2f}"


def print_run(result):
    if "error" in result:
        print(f"{result['case']} #{result['run']}: 실패 {result['error']}", flush=True)
        return
    print(
        f"{result['case']} #{result['run']}: tIoU {result['tiou']:.2f}, 정답률 {result['recall']:.2f}, "
        f"찾은 구간 {result['found']}, {result['seconds']:.0f}초, "
        f"토큰 입력 {result['input_tokens']:,} / 출력 {result['output_tokens']:,}",
        flush=True,
    )


def new_result_dir(runner, model, name):
    """results/<시각>_<runner>_<모델> 폴더. name 을 주면 그 이름을 쓴다."""
    if name == "":
        name = f"{datetime.now():%Y%m%d-%H%M}_{runner}_{model.replace(':', '-')}"
    result_dir = RESULTS_DIR / name
    result_dir.mkdir(parents=True)
    return result_dir


def save_report(result_dir, report, summary):
    (result_dir / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (result_dir / "summary.md").write_text(summary, encoding="utf-8")
    print()
    print(summary)
    print(f"기록: {result_dir}")
