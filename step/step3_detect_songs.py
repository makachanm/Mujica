"""
노래·음악 구간을 찾는다.

Whisper 는 노래 가사를 받아적거나, 배경음악 위에서 "시청해 주셔서 감사합니다" 같은 문장을 지어낸다.
그래서 노래·음악이 이어진 구간(songs)을 찾아 윈도우에 표시하고, 받아쓰기 전에 무음 처리할 구간(mute)을 정한다.

PANNs Cnn14 (AudioSet 527 클래스, 32kHz) 로 1초마다 말 / 노래 / 음악 확률을 뽑는다.
AudioSet 에는 Speech, Singing, Music 이 따로 있어서 말과 노래(가창)와 반주·배경음악을 나눠 볼 수 있다.

1) measure_sound: 무거운 신경망 단계. 1초마다 확률을 뽑아 캐시에 저장한다.
2) find_song_ranges: 가벼운 규칙 단계. 기준값, 스무딩, 최소 길이로 구간을 만든다.
"""

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import CACHE_DIR


# 공식 배포처(Zenodo)가 막힌 환경이 있어 같은 파일의 Hugging Face 사본을 쓴다.
MODEL_PATH = CACHE_DIR / "panns" / "Cnn14_32k.pth"
MODEL_URL = "https://huggingface.co/thelou1s/panns-inference/resolve/main/Cnn14_mAP%3D0.431.pth"
MODEL_SIZE = 327_428_481
LABELS_CSV_PATH = Path.home() / "panns_data" / "class_labels_indices.csv"
LABELS_CSV_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"

SAMPLE_RATE = 32000
STEP_SECONDS = 1.0       # 판정 간격
VIEW_SECONDS = 2.0       # 한 번 판정할 때 듣는 길이
# CPU 에서도 빨리 돌도록 합성곱 특징은 64초 블록마다 한 번만 계산하고,
# 그 위에서 2초 창을 1초씩 밀며 분류 헤드만 다시 돌린다.
BLOCK_SECONDS = 64.0
CONTEXT_SECONDS = 2.0    # 블록 경계 왜곡을 줄이려고 앞뒤로 더 읽는 길이
FEATURE_SECONDS = 0.32   # Cnn14 마지막 특징 한 칸의 시간 (10ms × 32배 풀링)

# AudioSet 클래스 번호 (class_labels_indices.csv 기준)
SPEECH_CLASSES = [0, 1, 2, 3, 4, 5]               # Speech, Male/Female/Child speech, Conversation, Narration
SINGING_CLASSES = [27, 28, 32, 33, 34, 35, 36]    # Singing, Choir, Male/Female/Child/Synthetic singing, Rapping
MUSIC_CLASSES = [137]                             # Music

SPEECH = 0
SINGING = 1
MUSIC = 2
SILENCE = 3
LABEL_COUNT = 4

SILENCE_DB = -50              # 이보다 조용하면 무음
SPEECH_THRESHOLD = 0.5        # 말소리 확률이 이 이상이면 음악이 깔려 있어도 말
SINGING_THRESHOLD = 0.3       # 노래 확률이 이 이상이고 말소리 확률보다 높으면 노래
MUSIC_THRESHOLD = 0.5         # 말소리가 약할 때 음악 확률이 이 이상이면 음악
SMOOTH_SECONDS = 5            # 짧게 튀는 판정을 다수결로 정리하는 창
SONG_MIN_SECONDS = 30         # 노래·음악이 이만큼 이어져야 노래 구간으로 본다
SONG_GAP_SECONDS = 5          # 노래 사이에 낀 이 정도의 짧은 말·무음은 같은 노래로 본다
KEEP_SPEECH_THRESHOLD = 0.15  # 노래 구간 안이라도 말소리 확률이 이 이상이면 받아쓴다 (곡 위의 말)
KEEP_PAD_SECONDS = 3          # 그 말 앞뒤로 남길 여유
MUTE_MIN_SECONDS = 5          # 이보다 짧은 노래 조각은 무음 처리하지 않는다

# 노래 구간을 "노래"(보컬 있음)와 "음악·BGM"으로 나누는 기준. 반주가 큰 곡에서는 노래 확률이
# 0.1~0.2 에 머물러 SINGING_THRESHOLD 로는 보컬을 못 잡는다. 가사가 받아써진 어쿠스틱 곡은 이 값 이상인
# 1초가 66%, 말 뒤 BGM·기타 전주는 0~3% 였다. 이 값 이상인 1초가 30% 이상이면 "노래"로 본다.
VOCAL_THRESHOLD = 0.08
VOCAL_SHARE = 0.3


def detect_songs(video_path):
    """{"songs": [{start, end, type}], "mute": [{start, end}]} 를 돌려준다. type 은 singing 또는 music."""
    sound = measure_sound(video_path)
    return find_song_ranges(sound)


def measure_sound(video_path):
    """1초마다 말 / 노래 / 음악 확률과 소리 크기. 같은 영상을 다시 돌릴 때는 캐시를 읽는다."""
    video_path = Path(video_path)
    cache_path = CACHE_DIR / f"{video_path.stem}_{video_path.stat().st_size}.npz"
    if cache_path.exists():
        saved = np.load(cache_path)
        return {name: saved[name] for name in saved.files}

    sound = classify_every_second(video_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = cache_path.with_suffix(".part.npz")
    np.savez_compressed(partial_path, **sound)
    partial_path.replace(cache_path)
    return sound


def load_model():
    download_model_files()
    # panns_inference 는 불러올 때 라벨 파일이 없으면 wget 을 부르므로, 파일을 받은 뒤에 불러온다.
    from panns_inference.models import Cnn14

    model = Cnn14(
        sample_rate=SAMPLE_RATE, window_size=1024, hop_size=320,
        mel_bins=64, fmin=50, fmax=14000, classes_num=527,
    )
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def download_model_files():
    if not LABELS_CSV_PATH.exists():
        LABELS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LABELS_CSV_URL, LABELS_CSV_PATH)

    if MODEL_PATH.exists():
        if MODEL_PATH.stat().st_size == MODEL_SIZE:
            return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial_path = MODEL_PATH.with_suffix(".part")
    print(f"[노래 감지] PANNs Cnn14 가중치 다운로드 (327MB) → {MODEL_PATH}", file=sys.stderr)
    urllib.request.urlretrieve(MODEL_URL, partial_path)
    if partial_path.stat().st_size != MODEL_SIZE:
        partial_path.unlink()
        raise RuntimeError("PANNs 가중치 크기가 예상과 다릅니다. 다운로드를 다시 시도하세요.")
    partial_path.replace(MODEL_PATH)


def classify_every_second(video_path):
    model = load_model()
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    # 영상 전체 소리를 한 번에 메모리에 올리지 않고, 블록에 필요한 만큼만 buffer 에 들고 있는다.
    buffer = np.zeros(0, dtype=np.float32)
    buffer_start = 0.0
    block_start = 0.0
    rows = []
    started_at = time.time()

    with torch.inference_mode():
        while True:
            buffer = read_audio_until(ffmpeg, buffer, buffer_start, block_start + BLOCK_SECONDS + CONTEXT_SECONDS)
            buffer_end = buffer_start + len(buffer) / SAMPLE_RATE
            if block_start >= buffer_end - 0.5:
                break

            block_end = min(block_start + BLOCK_SECONDS, buffer_end)
            rows += classify_block(model, buffer, buffer_start, block_start, block_end)

            if int(block_end) % 1800 < BLOCK_SECONDS:
                speed = block_end / max(1e-6, time.time() - started_at)
                print(f"[노래 감지] {block_end / 3600:5.2f}시간 처리, 실시간의 {speed:.0f}배", file=sys.stderr)

            block_start = block_end
            dropped_samples = int(max(0.0, block_start - CONTEXT_SECONDS - buffer_start) * SAMPLE_RATE)
            buffer = buffer[dropped_samples:]
            buffer_start += dropped_samples / SAMPLE_RATE

    ffmpeg.wait()
    if len(rows) == 0:
        raise RuntimeError(f"오디오를 읽지 못했습니다: {video_path}")

    table = np.array(rows, dtype=np.float32)
    print(f"[노래 감지] {len(table)}초 분류 완료 ({time.time() - started_at:.0f}s)", file=sys.stderr)
    return {
        "time": table[:, 0],
        "speech": table[:, 1],
        "singing": table[:, 2],
        "music": table[:, 3],
        "loudness_db": table[:, 4],
    }


def read_audio_until(ffmpeg, buffer, buffer_start, until):
    """buffer 가 until 초까지 찰 때까지 ffmpeg 출력을 더 읽는다. 소리가 끝나면 있는 만큼만 돌려준다."""
    while buffer_start + len(buffer) / SAMPLE_RATE < until:
        data = ffmpeg.stdout.read(SAMPLE_RATE * 4 * 16)
        if not data:
            break
        buffer = np.concatenate([buffer, np.frombuffer(data, dtype=np.float32)])
    return buffer


def classify_block(model, buffer, buffer_start, block_start, block_end):
    """[block_start, block_end) 를 1초마다 판정해 (시각, 말, 노래, 음악, 소리 크기) 줄 목록을 돌려준다."""
    audio_start = max(buffer_start, block_start - CONTEXT_SECONDS)
    audio = np.ascontiguousarray(buffer[int((audio_start - buffer_start) * SAMPLE_RATE):])
    features = convolution_features(model, audio)
    feature_centers = audio_start + (np.arange(features.shape[0]) + 0.5) * FEATURE_SECONDS
    seconds = np.arange(block_start, block_end, STEP_SECONDS)

    max_pooled = []
    mean_pooled = []
    for second in seconds:
        center = second + STEP_SECONDS / 2
        in_view = (feature_centers >= center - VIEW_SECONDS / 2) & (feature_centers < center + VIEW_SECONDS / 2)
        indexes = np.nonzero(in_view)[0]
        if len(indexes) == 0:
            nearest = min(int((center - audio_start) / FEATURE_SECONDS), features.shape[0] - 1)
            indexes = [nearest]
        view = features[indexes[0]:indexes[-1] + 1]
        max_pooled.append(view.max(dim=0).values)
        mean_pooled.append(view.mean(dim=0))

    # Cnn14 의 분류 헤드: (max + mean) 풀링 → fc1 → fc_audioset
    pooled = torch.stack(max_pooled) + torch.stack(mean_pooled)
    probabilities = torch.sigmoid(model.fc_audioset(F.relu_(model.fc1(pooled)))).numpy()

    rows = []
    for second, probability in zip(seconds, probabilities):
        first_sample = int((second - buffer_start) * SAMPLE_RATE)
        last_sample = int((second + STEP_SECONDS - buffer_start) * SAMPLE_RATE)
        samples = buffer[first_sample:last_sample]
        rms = 0.0
        if len(samples) > 0:
            rms = float(np.sqrt(np.mean(samples ** 2)))
        rows.append([
            second,
            probability[SPEECH_CLASSES].max(),
            probability[SINGING_CLASSES].max(),
            probability[MUSIC_CLASSES].max(),
            20 * np.log10(rms + 1e-9),
        ])
    return rows


def convolution_features(model, audio):
    """Cnn14 의 합성곱 부분만 통과시킨 시간축 특징 (특징 칸 수, 2048)."""
    x = torch.from_numpy(audio)[None, :]
    x = model.spectrogram_extractor(x)
    x = model.logmel_extractor(x)
    x = x.transpose(1, 3)
    x = model.bn0(x)
    x = x.transpose(1, 3)
    conv_blocks = [
        model.conv_block1, model.conv_block2, model.conv_block3,
        model.conv_block4, model.conv_block5, model.conv_block6,
    ]
    for block in conv_blocks[:-1]:
        x = block(x, pool_size=(2, 2), pool_type="avg")
    x = conv_blocks[-1](x, pool_size=(1, 1), pool_type="avg")
    x = torch.mean(x, dim=3)
    return x[0].T


def label_every_second(sound):
    """1초마다 SPEECH / SINGING / MUSIC / SILENCE 중 하나를 붙인다."""
    labels = np.full(len(sound["time"]), SPEECH, dtype=np.int64)
    is_singing = (sound["singing"] >= SINGING_THRESHOLD) & (sound["singing"] >= sound["speech"])
    is_music = (sound["music"] >= MUSIC_THRESHOLD) & (sound["speech"] < SPEECH_THRESHOLD) & ~is_singing
    is_silent = sound["loudness_db"] < SILENCE_DB
    labels[is_music] = MUSIC
    labels[is_singing] = SINGING
    labels[is_silent] = SILENCE
    return smooth_labels(labels)


def smooth_labels(labels):
    """SMOOTH_SECONDS 창 다수결로 짧게 튀는 판정을 정리한다. 동률이면 원래 값을 둔다."""
    half = SMOOTH_SECONDS // 2
    padded = np.concatenate([np.full(half, labels[0]), labels, np.full(half, labels[-1])])
    one_hot = np.zeros((len(padded), LABEL_COUNT), dtype=np.int32)
    one_hot[np.arange(len(padded)), padded] = 1
    running_total = np.vstack([np.zeros((1, LABEL_COUNT), dtype=np.int32), np.cumsum(one_hot, axis=0)])
    votes = (running_total[SMOOTH_SECONDS:] - running_total[:-SMOOTH_SECONDS])[:len(labels)]

    winner = votes.argmax(axis=1)
    original_is_tied = votes[np.arange(len(labels)), labels] == votes.max(axis=1)
    return np.where(original_is_tied, labels, winner)


def runs(values, times):
    """같은 값이 이어지는 구간들을 (값, 시작, 끝) 으로 돌려준다."""
    cuts = np.nonzero(np.diff(values))[0] + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(values)]
    return [(int(values[a]), float(times[a]), float(times[b - 1] + STEP_SECONDS)) for a, b in zip(starts, ends)]


def find_song_ranges(sound):
    """
    songs: 노래·음악 판정이 SONG_MIN_SECONDS 이상 이어진 곳. 그보다 짧은 건 말 사이 효과음으로 보고 넘어간다.
    mute:  songs 에서 말소리 부분을 뺀 곳. 곡 위에 말을 얹는 영상에서 그 말까지 지우지 않기 위해서다.
    """
    times = sound["time"]
    labels = label_every_second(sound)
    songlike = np.isin(labels, [SINGING, MUSIC]).astype(np.int64)

    joined = []
    for is_songlike, start, end in runs(songlike, times):
        if not is_songlike:
            continue
        if len(joined) > 0:
            if start - joined[-1]["end"] <= SONG_GAP_SECONDS:
                joined[-1]["end"] = end
                continue
        joined.append({"start": start, "end": end})

    songs = []
    in_song = np.zeros(len(times), dtype=bool)
    for song in joined:
        if song["end"] - song["start"] < SONG_MIN_SECONDS:
            continue
        inside = (times >= song["start"]) & (times < song["end"])
        vocal_share = np.mean(sound["singing"][inside] >= VOCAL_THRESHOLD)
        song_type = "music"
        if vocal_share >= VOCAL_SHARE:
            song_type = "singing"
        songs.append({"start": song["start"], "end": song["end"], "type": song_type})
        in_song |= inside

    talking = sound["speech"] >= KEEP_SPEECH_THRESHOLD
    pad_window = np.ones(2 * KEEP_PAD_SECONDS + 1, dtype=np.int64)
    # "same" 은 영상이 창보다 짧으면 창 길이만큼 돌려주므로, "full" 에서 가운데를 잘라 쓴다.
    padded_talking = np.convolve(talking.astype(np.int64), pad_window, "full")
    near_talking = padded_talking[KEEP_PAD_SECONDS:KEEP_PAD_SECONDS + len(times)] > 0

    mute = []
    for is_muted, start, end in runs((in_song & ~near_talking).astype(np.int64), times):
        if not is_muted:
            continue
        if end - start < MUTE_MIN_SECONDS:
            continue
        mute.append({"start": start, "end": end})

    return {"songs": songs, "mute": mute}
