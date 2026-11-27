"""
조각의 음성을 받아쓰고, 받아쓴 말을 윈도우마다 나눠 담는다.

윈도우마다 따로 받아쓰면 경계에서 말이 잘리고 Whisper 가 앞뒤 맥락 없이 10초씩만 듣는다.
그래서 조각을 길게(PIECE_SECONDS) 잘라 받아쓰고, 서버가 돌려준 세그먼트(시각 + 신뢰도)를
시작 시각이 속한 윈도우에 넣는다. 신뢰도로 Whisper 가 지어낸 문장을 걸러 낸다.

노래·음악 구간(step3)은 받아쓰기 전에 무음 처리하고, 그 구간에 걸친 윈도우에는 표시를 붙인다.
"""

import re
import sys
from pathlib import Path

import httpx

from common import run_command


SERVER_TIMEOUT_SECONDS = 600
PIECE_SECONDS = 600.0
# 덩어리 끝에서 말이 잘리지 않도록 조금 더 받아쓴다. 이 여유분에서 시작한 말은 다음 덩어리 몫이다.
TAIL_PAD_SECONDS = 5.0

# 무음 처리하고 남은 소리가 이보다 짧으면 서버에 보내지 않는다.
MIN_AUDIBLE_SECONDS = 1.0
# speaches 0.8.3 은 30초 이상인 오디오에 말소리가 없으면 500 을 낸다 (무음, 노래만 있는 구간 등).
# 500 이 나면 반씩 나눠 다시 요청하고, 이 길이보다 짧아져도 실패하면 그 부분은 건너뛴다.
MIN_SPLIT_SECONDS = 20.0

SONG_MARKS = {"singing": "(노래 구간)", "music": "(음악·BGM 구간)"}

# Whisper 가 지어낸 문장을 거르는 기준
MAX_COMPRESSION_RATIO = 2.4   # 같은 말을 되풀이함
MIN_AVG_LOGPROB = -1.0        # 신뢰도가 낮음 (가사나 잡음을 잘못 받아적은 경우가 대부분)
MAX_NO_SPEECH_PROB = 0.6      # 말소리가 없을 확률이 높음 (신뢰도도 -0.5 보다 낮을 때만)
MAX_REPEAT = 2                # 같은 문장이 이보다 많이 이어지면 나머지는 버린다
# 한국어 Whisper 가 무음이나 음악에서 자주 지어내는 문장. 실제로 자주 하는 말인 "구독 감사합니다"는 넣지 않는다.
MADE_UP_PATTERNS = [
    re.compile(pattern) for pattern in [
        r"시청\s*해\s*주셔서\s*감사합니다",
        r"구독과\s*좋아요",
        r"좋아요와\s*구독",
        r"자막\s*(제공|제작)",
        r"한글\s*자막",
        r"MBC\s*뉴스",
        r"다음\s*(영상|시간)에서?\s*(만나요|뵙겠습니다)",
        r"끝까지\s*시청",
    ]
]


class WhisperServerError(RuntimeError):
    def __init__(self, status_code, body):
        super().__init__(f"받아쓰기 서버 오류 ({status_code}): {body}")
        self.status_code = status_code


class WhisperServer:
    """OpenAI 호환 받아쓰기 서버(/v1/audio/transcriptions)에 맡긴다. speaches, LocalAI, vLLM 등."""

    def __init__(self, server_url, model_name, language):
        self.endpoint = server_url.rstrip("/") + "/v1/audio/transcriptions"
        self.model_name = model_name
        self.language = language

    def transcribe(self, audio_path):
        """세그먼트 목록을 돌려준다. 시각은 audio_path 파일 기준 초."""
        # 고유명사 사전은 hotwords 로 보내지 않는다. speaches + large-v3-turbo 에서 hotwords 를 주면
        # 고유명사는 고쳐지지만 받아쓰기 일부가 통째로 빠졌다 (25초 123자 → 20자).
        data = {
            "model": self.model_name,
            "language": self.language,
            "response_format": "verbose_json",
            "timestamp_granularities[]": "segment",
            "temperature": "0",
            # speaches 는 기본값이 true 라서, 보내지 않으면 세그먼트가 30초 단위로 뭉개진다.
            "without_timestamps": "false",
            "vad_filter": "true",
        }
        with open(audio_path, "rb") as audio_file:
            response = httpx.post(
                self.endpoint,
                files={"file": (Path(audio_path).name, audio_file, "audio/wav")},
                data=data,
                timeout=SERVER_TIMEOUT_SECONDS,
            )
        if response.status_code != 200:
            raise WhisperServerError(response.status_code, response.text[:500])

        body = response.json()
        if "segments" not in body:
            raise RuntimeError(f"받아쓰기 서버가 세그먼트를 돌려주지 않았습니다 (verbose_json 미지원). 응답 키: {sorted(body)}")

        segments = []
        for segment in body["segments"]:
            text = segment.get("text", "").strip()
            if text == "":
                continue
            segments.append({
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": text,
                "avg_logprob": segment.get("avg_logprob", 0.0),
                "no_speech_prob": segment.get("no_speech_prob", 0.0),
                "compression_ratio": segment.get("compression_ratio", 0.0),
            })
        return segments


def transcribe_chunk(video_path, chunk, work_dir, whisper_server, song_ranges):
    """조각을 받아써서 윈도우마다 transcript 를 채운다. song_ranges 는 step3 의 detect_songs 결과."""
    mute = song_ranges["mute"]
    segments = []
    piece_start = chunk.start
    piece_number = 0

    while piece_start < chunk.end:
        piece_end = min(piece_start + PIECE_SECONDS, chunk.end)
        read_end = piece_end
        if piece_end < chunk.end:
            read_end = piece_end + TAIL_PAD_SECONDS

        previous_end = -1.0
        if len(segments) > 0:
            previous_end = segments[-1]["end"]

        name = f"chunk{chunk.number:02d}_piece{piece_number:03d}"
        for segment in transcribe_piece(video_path, piece_start, read_end, work_dir, whisper_server, mute, name):
            if segment["start"] >= piece_end:
                break
            if segment["start"] < previous_end:
                if segment["end"] <= previous_end + 0.2:
                    continue  # 앞 덩어리의 꼬리 여유분에서 이미 받아쓴 말
            segments.append(segment)

        piece_start = piece_end
        piece_number += 1

    fill_windows(chunk, drop_made_up_segments(segments, mute), song_ranges["songs"])


def transcribe_piece(video_path, start, end, work_dir, whisper_server, mute, name):
    """[start, end) 를 받아쓴다. 세그먼트 시각은 영상 전체 기준 초로 돌려준다."""
    mute_in_piece = overlapping(mute, start, end)
    muted_seconds = sum(mute_end - mute_start for mute_start, mute_end in mute_in_piece)
    if end - start - muted_seconds < MIN_AUDIBLE_SECONDS:
        return []

    audio_path = str(Path(work_dir) / f"{name}.wav")
    arguments = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-t", f"{end - start:.3f}",
        "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000",
    ]
    if len(mute_in_piece) > 0:
        condition = "+".join(f"between(t,{mute_start:.3f},{mute_end:.3f})" for mute_start, mute_end in mute_in_piece)
        arguments += ["-af", f"volume=0:enable='{condition}'"]
    arguments.append(audio_path)
    run_command(arguments)

    try:
        segments = whisper_server.transcribe(audio_path)
    except WhisperServerError as error:
        if error.status_code != 500:
            raise
        if end - start < MIN_SPLIT_SECONDS * 2:
            print(f"[받아쓰기] 경고: {start:.1f}~{end:.1f}초 받아쓰기 실패 (500), 건너뜀", file=sys.stderr)
            return []
        middle = (start + end) / 2
        first_half = transcribe_piece(video_path, start, middle, work_dir, whisper_server, mute, f"{name}a")
        second_half = transcribe_piece(video_path, middle, end, work_dir, whisper_server, mute, f"{name}b")
        return first_half + second_half

    for segment in segments:
        segment["start"] = round(start + segment["start"], 2)
        segment["end"] = round(start + segment["end"], 2)
    return segments


def overlapping(ranges, start, end):
    """[start, end) 와 겹치는 구간들을 start 기준 상대 시각 (시작, 끝) 으로 잘라 돌려준다."""
    found = []
    for item in ranges:
        overlap_start = max(item["start"], start)
        overlap_end = min(item["end"], end)
        if overlap_end > overlap_start:
            found.append((overlap_start - start, overlap_end - start))
    return found


def drop_made_up_segments(segments, mute):
    """Whisper 가 지어냈거나 가사를 잘못 받아적은 세그먼트를 뺀다."""
    kept = []
    previous_text = None
    repeat_count = 0

    for segment in segments:
        normalized = re.sub(r"[\s.,!?~…]+", "", segment["text"])
        repeat_count += 1
        if normalized != previous_text:
            repeat_count = 1
        previous_text = normalized

        if looks_made_up(segment, normalized, repeat_count, mute):
            continue
        kept.append(segment)

    return kept


def looks_made_up(segment, normalized, repeat_count, mute):
    if normalized == "":
        return True
    if repeat_count > MAX_REPEAT:
        return True
    if segment["compression_ratio"] > MAX_COMPRESSION_RATIO:
        return True
    if segment["avg_logprob"] < MIN_AVG_LOGPROB:
        return True
    if segment["no_speech_prob"] > MAX_NO_SPEECH_PROB:
        if segment["avg_logprob"] < -0.5:
            return True

    for pattern in MADE_UP_PATTERNS:
        if pattern.search(segment["text"]):
            return True

    # 통째로 무음 처리한 구간 안에서 나온 말은 지어낸 것이다. 구간에 걸쳐 있는 말은 살린다.
    for item in mute:
        if item["start"] <= segment["start"]:
            if segment["end"] <= item["end"] + 0.5:
                return True

    return False


def fill_windows(chunk, segments, songs):
    """세그먼트를 시작 시각이 속한 윈도우에 담고, 노래·음악 구간과 절반 이상 겹치는 윈도우에는 표시를 붙인다."""
    for window in chunk.windows:
        texts = []
        for segment in segments:
            if window.start <= segment["start"] < window.end:
                texts.append(segment["text"])

        mark = find_song_mark(window, songs)
        if mark is not None:
            texts.insert(0, mark)

        window.transcript = " ".join(texts)


def find_song_mark(window, songs):
    for song in songs:
        overlap = min(song["end"], window.end) - max(song["start"], window.start)
        if overlap >= (window.end - window.start) / 2:
            return SONG_MARKS[song["type"]]
    return None
