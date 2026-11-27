import subprocess
from pathlib import Path


# 노래 감지 모델 가중치와 영상별 분석 결과를 둔다.
CACHE_DIR = Path.home() / ".cache" / "mujica"


class Window:
    def __init__(self, index, start, end):
        self.index = index
        self.start = start
        self.end = end
        self.frame_path = None
        self.transcript = ""


class Chunk:
    def __init__(self, number, start, end):
        self.number = number
        self.start = start
        self.end = end
        self.windows = []


class FoundSegment:
    def __init__(self, chunk_number, start, end, reason):
        self.chunk_numbers = [chunk_number]
        self.start = start
        self.end = end
        self.reasons = [reason]
        self.clip_path = None


def run_command(arguments):
    finished = subprocess.run(arguments, capture_output=True, text=True)
    if finished.returncode != 0:
        raise RuntimeError(f"{arguments[0]} 실패 (종료 코드 {finished.returncode}):\n{finished.stderr.strip()}")
    return finished.stdout


def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int(seconds % 3600 // 60)
    return f"{hours:02d}:{minutes:02d}:{seconds % 60:04.1f}"
