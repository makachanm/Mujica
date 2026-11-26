"""세그먼트 단위 받아쓰기 → 할루시네이션 필터 → Mujica 윈도우에 나눠 담기.

Mujica 원래 방식(step4)은 10초 윈도우마다 따로 받아쓰고 글자만 받는다. 그러면
- 윈도우 경계에서 말이 잘리고, Whisper가 앞뒤 맥락 없이 10초씩 듣는다.
- avg_logprob / compression_ratio / no_speech_prob 가 없어 할루시네이션을 거를 수 없다.

여기서는
1) 조각을 piece_seconds(기본 10분) 단위로 길게 잘라 받아쓰고 (노래 구간은 ffmpeg로 무음 처리),
2) 서버에서 verbose_json 으로 세그먼트(시각 + 신뢰도)를 받아,
3) addon/hallucination.py 로 거른 뒤,
4) 세그먼트 시작 시각이 속한 윈도우의 transcript 에 넣는다.
노래·음악 구간에 걸친 윈도우에는 "(노래 구간)" / "(음악·BGM 구간)" 표시를 붙여
에이전트가 "노래 부르는 장면" 같은 요청에도 답할 수 있게 한다.

받아쓰기 서버는 OpenAI 호환(/v1/audio/transcriptions)이어야 한다. speaches 기준으로
vad_filter, hotwords 도 같이 보낸다 (모르는 필드는 서버가 무시한다).
"""

import json
import subprocess
import sys
from pathlib import Path

import httpx

from addon.hallucination import DEFAULT_FILTER, filter_segments


SERVER_TIMEOUT_SECONDS = 600
PIECE_SECONDS = 600.0
# 조각 끝에서 말이 잘리지 않도록 뒤로 조금 더 보낸다. 이 구간에서 "시작한" 세그먼트는 다음 조각 몫이다.
TAIL_PAD_SECONDS = 5.0

SONG_MARKS = {"singing": "(노래 구간)", "music": "(음악·BGM 구간)"}

# 무음 처리하고 남은 소리가 이보다 짧으면 서버에 보내지 않는다
MIN_AUDIBLE_SECONDS = 1.0
# Speaches 0.8.3 은 30초 이상인 오디오에 말소리가 없으면 500 을 낸다 (무음·사인파·노래만 있는 구간에서 재현).
# 500 이 나면 구간을 반씩 나눠 다시 요청하고, 이 길이보다 짧아져도 실패하면 그 부분은 건너뛴다.
MIN_SPLIT_SECONDS = 20.0


class WhisperServerError(RuntimeError):
    def __init__(self, status_code, body):
        super().__init__(f"받아쓰기 서버 오류 ({status_code}): {body}")
        self.status_code = status_code


class SegmentWhisperServer:
    """OpenAI 호환 받아쓰기 서버에 verbose_json 으로 요청해 세그먼트 목록을 받는다."""

    def __init__(self, server_url, model_name, language, hotwords=None, vad_filter=True):
        self.endpoint = server_url.rstrip("/") + "/v1/audio/transcriptions"
        self.model_name = model_name
        self.language = language
        self.hotwords = hotwords
        self.vad_filter = vad_filter

    def transcribe(self, audio_path):
        data = {
            "model": self.model_name,
            "language": self.language,
            "response_format": "verbose_json",
            "timestamp_granularities[]": "segment",
            "temperature": "0",
            # Speaches 는 기본값이 true 라서, 보내지 않으면 세그먼트가 30초 디코딩 단위로 뭉개진다
            "without_timestamps": "false",
            "vad_filter": "true" if self.vad_filter else "false",
        }
        if self.hotwords:
            data["hotwords"] = self.hotwords

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
            raise RuntimeError(
                "받아쓰기 서버가 세그먼트를 돌려주지 않았습니다 (verbose_json 미지원). "
                f"응답 키: {sorted(body)}"
            )

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


def overlapping(ranges, start, end):
    """[start, end) 와 겹치는 구간들을 start 기준 상대 시각으로 잘라 돌려준다."""
    found = []
    for item in ranges:
        a = max(item["start"], start)
        b = min(item["end"], end)
        if b > a:
            found.append((a - start, b - start))
    return found


def extract_piece(video_path, start, duration, mute_ranges, audio_path):
    """[start, start+duration) 음성을 16kHz mono wav 로 뽑고, mute_ranges(상대 시각)는 무음으로 바꾼다."""
    arguments = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
    ]
    if mute_ranges:
        condition = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in mute_ranges)
        arguments += ["-af", f"volume=0:enable='{condition}'"]
    arguments.append(str(audio_path))

    finished = subprocess.run(arguments, capture_output=True, text=True)
    if finished.returncode != 0:
        raise RuntimeError(f"ffmpeg 실패 (종료 코드 {finished.returncode}):\n{finished.stderr.strip()}")


def audible_seconds(mute, start, end):
    muted = sum(b - a for a, b in overlapping(mute, start, end))
    return (end - start) - muted


def transcribe_piece(video_path, start, end, work_dir, server, mute, name):
    """[start, end) 한 덩어리를 받아쓴다. 세그먼트 시각은 영상 전체 기준 초로 돌려준다.

    다 무음 처리된 덩어리는 보내지 않고, 서버가 500 을 내면 반씩 나눠 다시 시도한다.
    """
    if audible_seconds(mute, start, end) < MIN_AUDIBLE_SECONDS:
        return []

    audio_path = Path(work_dir) / f"{name}.wav"
    extract_piece(video_path, start, end - start, overlapping(mute, start, end), audio_path)
    try:
        segments = server.transcribe(audio_path)
    except WhisperServerError as error:
        if error.status_code != 500:
            raise
        if end - start < MIN_SPLIT_SECONDS * 2:
            print(f"[stt] 경고: {start:.1f}~{end:.1f}초 받아쓰기 실패 (500), 건너뜀", file=sys.stderr)
            return []
        middle = (start + end) / 2
        return (transcribe_piece(video_path, start, middle, work_dir, server, mute, f"{name}a")
                + transcribe_piece(video_path, middle, end, work_dir, server, mute, f"{name}b"))

    for segment in segments:
        segment["start"] = round(start + segment["start"], 2)
        segment["end"] = round(start + segment["end"], 2)
    return segments


def transcribe_range(video_path, start, end, work_dir, server, mute=None, piece_seconds=PIECE_SECONDS, name="range"):
    """[start, end) 를 piece_seconds 씩 받아쓴 세그먼트 목록 (영상 전체 기준 초)."""
    mute = mute or []
    pieces = []
    piece_start = start
    while piece_start < end:
        piece_end = min(piece_start + piece_seconds, end)
        pieces.append((piece_start, piece_end))
        piece_start = piece_end

    segments = []
    for order, (piece_start, piece_end) in enumerate(pieces):
        is_last = order == len(pieces) - 1
        read_end = piece_end if is_last else piece_end + TAIL_PAD_SECONDS
        previous_end = segments[-1]["end"] if segments else -1.0
        piece_segments = transcribe_piece(
            video_path, piece_start, read_end, work_dir, server, mute, f"{name}_piece{order:03d}",
        )
        for segment in piece_segments:
            if segment["start"] >= piece_end and not is_last:
                break  # 꼬리 여유분에서 시작한 말은 다음 조각이 처리한다
            if segment["start"] < previous_end and segment["end"] <= previous_end + 0.2:
                continue  # 앞 조각의 꼬리 여유분이 이미 덮은 말
            segments.append(segment)

    return segments


def song_mark(window, songs):
    """윈도우가 노래·음악 구간과 절반 이상 겹치면 그 표시를 돌려준다."""
    length = window.end - window.start
    for song in songs:
        overlap = min(song["end"], window.end) - max(song["start"], window.start)
        if length > 0 and overlap >= length / 2:
            return SONG_MARKS.get(song.get("type"), SONG_MARKS["music"])
    return None


def fill_windows(chunk, segments, songs):
    """세그먼트를 시작 시각이 속한 윈도우에 담는다. 노래 구간 윈도우에는 표시를 붙인다."""
    texts = {window.index: [] for window in chunk.windows}
    for segment in segments:
        for window in chunk.windows:
            if window.start <= segment["start"] < window.end:
                texts[window.index].append(segment["text"])
                break

    for window in chunk.windows:
        transcript = " ".join(texts[window.index])
        mark = song_mark(window, songs)
        if mark is not None:
            transcript = f"{mark} {transcript}".strip()
        window.transcript = transcript


def transcribe_chunk(video_path, chunk, work_dir, server, song_info=None, filter_settings=None,
                     piece_seconds=PIECE_SECONDS):
    """Mujica step4 transcribe_windows 를 대신한다. 윈도우마다 transcript 를 채우고 걸러낸 결과를 돌려준다.

    song_info: addon.song_detect.detect_songs() 결과. None 이면 노래 처리를 하지 않는다.
    돌려주는 값: {"kept": 남긴 세그먼트 수, "dropped": {이유: 개수}}
    """
    song_info = song_info or {"songs": [], "mute": []}
    mute = song_info["mute"]

    segments = transcribe_range(
        video_path, chunk.start, chunk.end, work_dir, server,
        mute=mute, piece_seconds=piece_seconds, name=f"chunk{chunk.number:02d}",
    )
    kept, dropped = filter_segments(segments, {**DEFAULT_FILTER, **(filter_settings or {})}, mute)
    fill_windows(chunk, kept, song_info["songs"])

    reasons = {}
    for segment in dropped:
        reasons[segment["reason"]] = reasons.get(segment["reason"], 0) + 1

    # 무엇을 버렸는지 나중에 확인할 수 있게 남겨 둔다 (work_dir 가 임시 폴더면 같이 지워진다)
    log_path = Path(work_dir) / f"chunk{chunk.number:02d}_stt_dropped.json"
    log_path.write_text(json.dumps(dropped, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"kept": len(kept), "dropped": reasons}
